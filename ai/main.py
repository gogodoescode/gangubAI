"""GangubAI entrypoint.

    python3 -m ai.main --fullscreen    face in front + voice loop in the background
    python3 -m ai.main --voice-only    just the voice loop (no window)

Voice loop: wake word -> record -> whisper.cpp -> (robot keyword shortcut | LangGraph brain)
-> Piper TTS, with emotion events sent to the face over localhost UDP.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
from typing import Any

from ai import config
from ai.face import send_to_face

RAG_LOOKUP_MESSAGE = "Let me look into your slides and get back to you."
THREAD_ID = "voice-session"
BACKEND_LOG = "voice_backend.log"


def emit_emotion(emotion: str, event: str) -> None:
    send_to_face(emotion, event=event)
    print(f"[Emotion] {emotion} ({event})", flush=True)


# ── Robot keyword shortcut (skips the LLM for clear movement commands) ──

def _extract_duration_seconds(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", text)
    return max(0.2, min(10.0, float(match.group(1)))) if match else None


def handle_robot_intent(user_text: str, brain) -> tuple[str, str] | None:
    """Return (reply, emotion) if the text is a movement/wander command, else None."""
    text = user_text.strip().lower()

    if "wander" in text:
        action = "stop" if any(k in text for k in ("stop", "off", "disable")) else "start"
        return brain.set_wander(action), "excited" if action == "start" else "neutral"

    if any(k in text for k in ("spin", "rotate", "360", "turn around")):
        direction = "360"
    elif "forward" in text or "ahead" in text:
        direction = "forward"
    elif "backward" in text or "reverse" in text or re.search(r"\bback\b", text):
        direction = "backward"
    elif re.search(r"\bleft\b", text):
        direction = "left"
    elif re.search(r"\bright\b", text):
        direction = "right"
    elif re.search(r"\bstop\b", text):
        direction = "stop"
    else:
        return None

    return brain.move(direction, _extract_duration_seconds(text)), "neutral" if direction == "stop" else "excited"


# ── Chatbot turn ─────────────────────────────────────────────

def _text_of(content: Any) -> str:
    """Extract plain text from LangChain message content (str or list of parts)."""
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else p.get("text", "") for p in content if isinstance(p, (str, dict))]
        return " ".join(p.strip() for p in parts if p and p.strip())
    return str(content).strip()


def _requested_retrieval(state: dict[str, Any]) -> bool:
    """True if the latest AI message asked for retrieve_context."""
    from langchain_core.messages import AIMessage

    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return any(call.get("name") == "retrieve_context" for call in message.tool_calls or [])
    return False


def chatbot_turn(chatbot, user_text: str, on_retrieval_start) -> tuple[str, str]:
    """Run one chatbot turn and return (assistant_text, emotion)."""
    from langchain_core.messages import HumanMessage

    final_state: dict[str, Any] = {}
    notified = False
    for state in chatbot.stream(
        {"messages": [HumanMessage(content=user_text)]},
        config={"configurable": {"thread_id": THREAD_ID}},
        stream_mode="values",
    ):
        if not notified and _requested_retrieval(state):
            notified = True
            on_retrieval_start()
        final_state = state

    messages = final_state.get("messages", [])
    text = _text_of(messages[-1].content) if messages else ""
    return text, str(final_state.get("current_emotion", "neutral"))


# ── Voice loop ───────────────────────────────────────────────

def run_voice_loop() -> int:
    from langgraph.checkpoint.memory import MemorySaver

    from ai import brain
    from ai.voice import TTSWorker, WakeWordDetector, record_until_silence, transcribe

    print("=" * 64)
    print("GangubAI Voice Loop")
    print("=" * 64)

    detector = WakeWordDetector()
    tts = TTSWorker()
    tts.start()
    chatbot = brain.graph.compile(checkpointer=MemorySaver())

    def say(text: str, emotion: str) -> None:
        print(f"[Assistant] {text}", flush=True)
        emit_emotion(emotion, "speech_start")
        if tts.enqueue(text) and not tts.wait_until_done(timeout_s=config.TTS_TIMEOUT_S):
            print(f"[TTS] Timeout after {config.TTS_TIMEOUT_S:.0f}s; continuing.", flush=True)
        emit_emotion("happy", "speech_end")

    def on_retrieval_start() -> None:
        print(f"[Assistant] {RAG_LOOKUP_MESSAGE}", flush=True)
        emit_emotion("thinking", "speech_start")
        tts.enqueue(RAG_LOOKUP_MESSAGE)

    try:
        while True:
            detector.wait_for_wake_word()
            # Always stop wandering when the user calls the robot.
            brain.set_wander("stop", check_subscribers=False)

            wav_path = record_until_silence()
            if not wav_path:
                print("[Main] No audio captured.", flush=True)
                continue

            try:
                user_text = transcribe(wav_path)
            except Exception as exc:
                print(f"[STT] Transcription error: {exc}", flush=True)
                continue
            if not user_text:
                print("[STT] Empty transcription.", flush=True)
                continue
            print(f"[User] {user_text}", flush=True)

            robot_reply = handle_robot_intent(user_text, brain)
            if robot_reply is not None:
                say(*robot_reply)
                continue

            try:
                reply_text, emotion = chatbot_turn(chatbot, user_text, on_retrieval_start)
            except Exception as exc:
                print(f"[Chatbot] Error: {exc}", flush=True)
                continue
            if reply_text:
                say(reply_text, emotion)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user. Shutting down...", flush=True)
        return 0
    finally:
        tts.stop()


# ── Face + voice together ────────────────────────────────────

def _tee_backend_output(proc: subprocess.Popen, log_path: str) -> None:
    """Mirror voice-loop output to the log file and this terminal."""
    with open(log_path, "a", encoding="utf-8") as log_file:
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(f"[Voice] {line}")
            sys.stdout.flush()


def run_face_and_voice(fullscreen: bool) -> int:
    from ai.face import FaceApp

    voice = subprocess.Popen(
        [sys.executable, "-m", "ai.main", "--voice-only"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    threading.Thread(target=_tee_backend_output, args=(voice, BACKEND_LOG), daemon=True).start()
    print(f"[Main] Voice loop started (pid={voice.pid}), logging to {BACKEND_LOG}", flush=True)

    try:
        return FaceApp(fullscreen=fullscreen).run()
    finally:
        voice.terminate()
        try:
            voice.wait(timeout=4)
        except subprocess.TimeoutExpired:
            voice.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="GangubAI robot")
    parser.add_argument("--fullscreen", action="store_true", help="Start the face in fullscreen")
    parser.add_argument("--voice-only", action="store_true", help="Run only the voice loop, no face window")
    args = parser.parse_args()
    if args.voice_only:
        return run_voice_loop()
    return run_face_and_voice(args.fullscreen)


if __name__ == "__main__":
    raise SystemExit(main())
