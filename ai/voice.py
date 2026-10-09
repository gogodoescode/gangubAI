"""GangubAI ears and mouth: wake word, recording, speech-to-text, text-to-speech.

- WakeWordDetector: OpenWakeWord ("Hi Gungu Bai")
- record_until_silence(): mic -> WAV, stops on silence
- transcribe(): whisper.cpp
- TTSWorker: Piper -> aplay (Pi + MAX98357A) or sounddevice (PC), set by config.TTS_USE_APLAY
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import IO

import numpy as np
import scipy.signal
import sounddevice as sd

from ai import config


def _input_samplerate(device: str | int | None = config.INPUT_DEVICE_NAME) -> int:
    try:
        return int(sd.query_devices(device=device, kind="input")["default_samplerate"])
    except Exception:
        return 48000


# ═══════════════════════════════════════════════════════════════════
# Wake word
# ═══════════════════════════════════════════════════════════════════

class WakeWordDetector:
    """Blocks on the microphone until the wake word is heard."""

    def __init__(self) -> None:
        from openwakeword.model import Model

        model_path = Path(config.WAKE_WORD_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(f"Wake word model not found: {model_path}. Set ai.config.WAKE_WORD_MODEL.")

        # Use local feature models when present, otherwise OpenWakeWord's bundled ones.
        model_kwargs: dict[str, str] = {}
        melspec_path = Path(config.WAKE_WORD_ASSETS_DIR) / "melspectrogram.onnx"
        embedding_path = Path(config.WAKE_WORD_ASSETS_DIR) / "embedding_model.onnx"
        if melspec_path.exists() and embedding_path.exists():
            model_kwargs = {"melspec_model_path": str(melspec_path), "embedding_model_path": str(embedding_path)}

        self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx", **model_kwargs)
        print(f"[WakeWord] Loaded wake-word model: {model_path}", flush=True)

    def wait_for_wake_word(self) -> None:
        """Listen on the microphone and return once the wake word is detected."""
        self._model.reset()

        # Record at the mic's native rate and resample each chunk to 16 kHz.
        native_rate = _input_samplerate()
        input_chunk = int(config.OWW_CHUNK_SIZE * native_rate / config.OWW_SAMPLE_RATE)

        print("[WakeWord] Listening for wake word...", flush=True)
        with sd.InputStream(
            samplerate=native_rate,
            channels=1,
            dtype="int16",
            blocksize=input_chunk,
            device=config.INPUT_DEVICE_NAME,
        ) as stream:
            while True:
                data, _ = stream.read(input_chunk)
                audio = data[:, 0]
                if native_rate != config.OWW_SAMPLE_RATE:
                    audio = scipy.signal.resample(audio, config.OWW_CHUNK_SIZE).astype(np.int16)

                scores = self._model.predict(audio)
                if max(scores.values(), default=0.0) >= config.WAKE_WORD_THRESHOLD:
                    self._model.reset()
                    return


# ═══════════════════════════════════════════════════════════════════
# Recording
# ═══════════════════════════════════════════════════════════════════

def record_until_silence(filename: str = config.RECORDING_OUTPUT_FILE) -> str | None:
    """Record until SILENCE_DURATION_S of silence (or MAX_RECORD_TIME_S).

    Returns the saved 16-bit mono WAV path, or None if nothing was captured.
    """
    time.sleep(config.PRE_RECORD_DELAY_S)

    samplerate = _input_samplerate()
    chunk_duration_s = 0.05
    silent_chunks_to_stop = int(config.SILENCE_DURATION_S / chunk_duration_s)
    max_chunks = int(config.MAX_RECORD_TIME_S / chunk_duration_s)

    buffer: list[np.ndarray] = []
    silent_chunks = 0
    stop_flag = threading.Event()

    def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        nonlocal silent_chunks
        buffer.append(indata.copy())
        if len(buffer) < 5:
            return  # ignore mic warm-up noise
        volume = float(np.linalg.norm(indata) / np.sqrt(max(len(indata), 1)))
        silent_chunks = silent_chunks + 1 if volume < config.SILENCE_THRESHOLD else 0
        if silent_chunks >= silent_chunks_to_stop:
            stop_flag.set()

    print("[Recorder] Listening...", flush=True)
    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=int(samplerate * chunk_duration_s),
            device=config.INPUT_DEVICE_NAME,
            callback=_callback,
        ):
            while not stop_flag.is_set() and len(buffer) < max_chunks:
                sd.sleep(int(chunk_duration_s * 1000))
    except Exception as exc:
        print(f"[Recorder] InputStream error: {exc}", flush=True)
        return None

    if not buffer:
        return None
    audio = np.nan_to_num(np.concatenate(buffer).flatten())
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())
    print(f"[Recorder] Saved {filename!r} ({len(audio_int16) / samplerate:.1f}s)", flush=True)
    return filename


# ═══════════════════════════════════════════════════════════════════
# Speech to text (whisper.cpp)
# ═══════════════════════════════════════════════════════════════════

_SEGMENT_RE = re.compile(r"^\[[0-9:.]+\s*-->\s*[0-9:.]+\]\s*(.*)$")


def transcribe(wav_path: str) -> str:
    """Transcribe a WAV file with whisper.cpp. Raises on missing files or failures."""
    for path, setting in ((config.WHISPER_CPP_BIN, "WHISPER_CPP_BIN"), (config.WHISPER_CPP_MODEL, "WHISPER_CPP_MODEL")):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Set ai.config.{setting}.")

    cmd = [
        config.WHISPER_CPP_BIN,
        "-m", config.WHISPER_CPP_MODEL,
        "-l", config.WHISPER_LANGUAGE,
        "-t", str(config.WHISPER_CPP_THREADS),
        "-f", wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.WHISPER_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"whisper.cpp timed out after {config.WHISPER_TIMEOUT_S}s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"whisper.cpp failed (exit {result.returncode}): {result.stderr.strip() or 'no stderr'}")

    # Output lines look like "[00:00:00.000 --> 00:00:02.000]  hello there".
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    segments = [m.group(1).strip() for line in lines if (m := _SEGMENT_RE.match(line)) and m.group(1).strip()]
    if segments:
        return " ".join(segments)
    return lines[-1] if lines else ""


# ═══════════════════════════════════════════════════════════════════
# Text to speech (Piper)
# ═══════════════════════════════════════════════════════════════════

_SANITIZE_RE = re.compile(r"[^\w\s,.!?:;'\-]")


def _sanitize_text_for_tts(text: str) -> str:
    """Strip characters Piper can't speak and collapse whitespace."""
    return re.sub(r"\s+", " ", _SANITIZE_RE.sub("", text or "")).strip()


def _chunk_text(text: str, max_chars: int = config.TTS_QUEUE_CHUNK_CHARS) -> list[str]:
    """Split text into word-safe chunks of at most *max_chars*."""
    chunks: list[str] = []
    current: list[str] = []
    for word in text.split():
        if current and len(" ".join(current + [word])) > max_chars:
            chunks.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _play_with_aplay(pcm: IO[bytes]) -> None:
    """Stream Piper S16_LE mono PCM to aplay as S32_LE stereo (MAX98357A format)."""
    aplay = subprocess.Popen(
        ["aplay", "-q", "-D", "default", "-f", "S32_LE", "-r", str(config.PIPER_SAMPLE_RATE), "-c", "2"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert aplay.stdin is not None and aplay.stderr is not None
    try:
        try:
            while raw := pcm.read(config.TTS_READ_CHUNK_BYTES):
                s32 = np.frombuffer(raw, dtype=np.int16).astype(np.int32) << 16
                aplay.stdin.write(np.repeat(s32, 2).tobytes())
            aplay.stdin.close()
        except BrokenPipeError:
            pass
        rc = aplay.wait(timeout=10)
        if rc != 0:
            msg = aplay.stderr.read().decode("utf-8", errors="ignore").strip() or "no stderr"
            raise RuntimeError(f"aplay failed (exit {rc}): {msg}")
    finally:
        aplay.stderr.close()
        if aplay.poll() is None:
            aplay.kill()


def _play_with_sounddevice(pcm: IO[bytes]) -> None:
    """Stream Piper PCM to a sounddevice output, resampling if the device needs it."""
    piper_rate = config.PIPER_SAMPLE_RATE
    device = config.OUTPUT_DEVICE_NAME
    try:
        sd.check_output_settings(device=device, samplerate=piper_rate, channels=1, dtype="int16")
        playback_rate = piper_rate
    except Exception:
        playback_rate = int(sd.query_devices(device=device, kind="output")["default_samplerate"])

    with sd.RawOutputStream(
        samplerate=playback_rate,
        channels=1,
        dtype="int16",
        device=device,
        latency="high",
        blocksize=config.TTS_STREAM_BLOCKSIZE,
    ) as stream:
        while raw := pcm.read(config.TTS_READ_CHUNK_BYTES):
            if playback_rate != piper_rate:
                audio = np.frombuffer(raw, dtype=np.int16)
                samples = max(1, int(len(audio) * playback_rate / piper_rate))
                raw = scipy.signal.resample(audio, samples).astype(np.int16).tobytes()
            stream.write(raw)
        # Short silence tail so the last word isn't clipped.
        stream.write(np.zeros(int(playback_rate * config.TTS_TAIL_SILENCE_S), dtype=np.int16).tobytes())


def speak(text: str) -> None:
    """Synthesize *text* with Piper and play it (blocking)."""
    utterance = _sanitize_text_for_tts(text)
    if not utterance:
        return

    for path, setting in ((config.PIPER_BIN, "PIPER_BIN"), (config.PIPER_VOICE_MODEL, "PIPER_VOICE_MODEL")):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Set ai.config.{setting}.")

    proc = subprocess.Popen(
        [config.PIPER_BIN, "--model", config.PIPER_VOICE_MODEL, "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        proc.stdin.write(utterance.encode("utf-8") + b"\n")
        proc.stdin.close()
        if config.TTS_USE_APLAY:
            _play_with_aplay(proc.stdout)
        else:
            _play_with_sounddevice(proc.stdout)
        proc.wait(timeout=2)
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()


class TTSWorker:
    """Background thread that speaks queued text chunks in order."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tts-worker")
        self._speaking = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=3)

    def enqueue(self, text: str) -> int:
        """Chunk and enqueue text. Returns number of queued chunks."""
        chunks = _chunk_text(_sanitize_text_for_tts(text))
        for chunk in chunks:
            self._queue.put(chunk)
        return len(chunks)

    def wait_until_done(self, timeout_s: float) -> bool:
        """Wait until the queue is drained and nothing is playing."""
        deadline = time.monotonic() + timeout_s
        while self._queue.unfinished_tasks or self._speaking.is_set():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            self._speaking.set()
            try:
                speak(item)
            except Exception as exc:
                print(f"[TTS] Error: {exc}", flush=True)
            finally:
                self._speaking.clear()
                self._queue.task_done()
