"""Piper TTS for the GangubAI voice pipeline.

Output backend is chosen by config.TTS_USE_APLAY:
- aplay: Raspberry Pi + MAX98357A I2S amp (needs S32_LE stereo)
- sounddevice: regular laptop / PC speakers
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import IO

import numpy as np
import scipy.signal
import sounddevice as sd

from ai.voice import config


_SANITIZE_RE = re.compile(r"[^\w\s,.!?:;'\-]")


def _sanitize_text_for_tts(text: str) -> str:
    """Strip characters Piper can't speak and collapse whitespace."""
    cleaned = _SANITIZE_RE.sub("", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


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


def _play_with_aplay(pcm: IO[bytes], piper_rate: int) -> None:
    """Stream Piper S16_LE mono PCM to aplay as S32_LE stereo (MAX98357A format)."""
    aplay = subprocess.Popen(
        ["aplay", "-q", "-D", "default", "-f", "S32_LE", "-r", str(piper_rate), "-c", "2"],
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


def _play_with_sounddevice(pcm: IO[bytes], piper_rate: int) -> None:
    """Stream Piper PCM to a sounddevice output, resampling if the device needs it."""
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
            raise FileNotFoundError(f"{path} not found. Set ai.voice.config.{setting}.")

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
            _play_with_aplay(proc.stdout, config.PIPER_SAMPLE_RATE)
        else:
            _play_with_sounddevice(proc.stdout, config.PIPER_SAMPLE_RATE)
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
        self.last_error: Exception | None = None

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
                self.last_error = exc
                print(f"[TTS] Error: {exc}", flush=True)
            finally:
                self._speaking.clear()
                self._queue.task_done()
