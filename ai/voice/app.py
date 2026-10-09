"""GangubAI voice loop.

Wake word -> record -> whisper.cpp -> (robot keyword shortcut | LangGraph chatbot)
-> Piper TTS, sending emotion events to the face over localhost UDP.

Usage:
    python3 -m ai.voice.app
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from ai import robot
from ai.frontend.bridge.local_queue import EmotionPublisher
from ai.voice import config
from ai.voice.recorder import record_adaptive
from ai.voice.transcriber import transcribe
from ai.voice.tts import TTSWorker
from ai.voice.wake_word import WakeWordDetector


RAG_LOOKUP_MESSAGE = "Let me look into your slides and get back to you."
THREAD_ID = "voice-session"
TTS_TIMEOUT_S = 180.0

_EMOTION_PUBLISHER = EmotionPublisher()


def emit_emotion(emotion: str, event: str) -> None:
    """Tell the face which emotion to show and whether speech started/ended."""
    normalized = str(emotion).strip().lower() or "neutral"
    _EMOTION_PUBLISHER.publish(normalized, source="voice", payload=json.dumps({"event": event}))
    print(f"[Emotion] {normalized} ({event})", flush=True)


# ── Robot keyword shortcut (skips the LLM for clear movement commands) ──

def _extract_duration_seconds(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", text)
    return max(0.2, min(10.0, float(match.group(1)))) if match else None


def _handle_robot_intent(user_text: str) -> tuple[str, str] | None:
    """Return (reply, emotion) if the text is a movement/wander command, else None."""
    text = user_text.strip().lower()

    if "wander" in text:
        action = "stop" if any(k in text for k in ("stop", "off", "disable")) else "start"
        return robot.set_wander(action), "excited" if action == "start" else "neutral"

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

    return robot.move(direction, _extract_duration_seconds(text)), "neutral" if direction == "stop" else "excited"


# ── Chatbot ──────────────────────────────────────────────────

def _text_of(content: Any) -> str:
    """Extract plain text from LangChain message content (str or list of parts)."""
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else p.get("text", "") for p in content if isinstance(p, (str, dict))]
        return " ".join(p.strip() for p in parts if p and p.strip())
    return str(content).strip()


def _requested_retrieval(state: dict[str, Any]) -> bool:
    """True if the latest AI message asked for retrieve_context."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return any(call.get("name") == "retrieve_context" for call in message.tool_calls or [])
    return False


def _chatbot_turn(chatbot, user_text: str, on_retrieval_start) -> tuple[str, str]:
    """Run one chatbot turn and return (assistant_text, emotion)."""
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


# ── Main loop ────────────────────────────────────────────────

def _say(tts: TTSWorker, text: str, emotion: str) -> None:
    print(f"[Assistant] {text}", flush=True)
    emit_emotion(emotion, event="speech_start")
    if tts.enqueue(text) and not tts.wait_until_done(timeout_s=TTS_TIMEOUT_S):
        print(f"[TTS] Timeout after {TTS_TIMEOUT_S:.0f}s; continuing.", flush=True)
    emit_emotion("happy", event="speech_end")


def main() -> int:
    print("=" * 64)
    print("GangubAI Voice App")
    print("=" * 64)

    detector = WakeWordDetector()
    tts = TTSWorker()
    tts.start()
    chatbot = None  # built on first non-robot query (loads the LLM + vector store)

    def on_retrieval_start() -> None:
        print(f"[Assistant] {RAG_LOOKUP_MESSAGE}", flush=True)
        emit_emotion("thinking", event="speech_start")
        tts.enqueue(RAG_LOOKUP_MESSAGE)

    try:
        while True:
            detector.wait_for_wake_word()
            # Always stop wandering when the user calls the robot.
            robot.set_wander("stop", check_subscribers=False)

            wav_path = record_adaptive(filename=config.RECORDING_OUTPUT_FILE)
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

            robot_reply = _handle_robot_intent(user_text)
            if robot_reply is not None:
                _say(tts, *robot_reply)
                continue

            if chatbot is None:
                from ai.chatbot.graph import graph
                chatbot = graph.compile(checkpointer=MemorySaver())

            try:
                reply_text, emotion = _chatbot_turn(chatbot, user_text, on_retrieval_start)
            except Exception as exc:
                print(f"[Chatbot] Error: {exc}", flush=True)
                continue

            if reply_text:
                _say(tts, reply_text, emotion)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user. Shutting down...", flush=True)
        return 0
    finally:
        tts.stop()


if __name__ == "__main__":
    raise SystemExit(main())
