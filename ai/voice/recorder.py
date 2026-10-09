"""Audio recorder for the GangubAI voice pipeline: records until silence is detected."""

import threading
import time
import wave

import numpy as np
import sounddevice as sd

from ai.voice import config


# ── Internal helpers ───────────────────────────────────────────────────────

def _get_input_samplerate() -> int:
    """Return the default input device's native sample rate."""
    try:
        info = sd.query_devices(kind="input")
        return int(info["default_samplerate"])
    except Exception:
        return 44100


def _save_buffer(chunks: list[np.ndarray], filename: str, samplerate: int) -> str | None:
    """Concatenate recorded chunks and write a 16-bit mono WAV file.

    Returns the file path on success, None if the buffer is empty.
    """
    if not chunks:
        return None

    audio = np.concatenate(chunks, axis=0).flatten()
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

    duration = len(audio_int16) / samplerate
    print(f"[Recorder] Saved {filename!r}  ({duration:.1f}s, {samplerate}Hz)", flush=True)
    return filename


# ── Public API ─────────────────────────────────────────────────────────────

def record_adaptive(
    filename: str = config.RECORDING_OUTPUT_FILE,
    silence_threshold: float = config.SILENCE_THRESHOLD,
    silence_duration_s: float = config.SILENCE_DURATION_S,
    max_record_time_s: float = config.MAX_RECORD_TIME_S,
    pre_record_delay_s: float = config.PRE_RECORD_DELAY_S,
    device: str | int | None = config.INPUT_DEVICE_NAME,
) -> str | None:
    """Record audio using Voice Activity Detection (VAD).

    Starts immediately and stops automatically once silence lasting
    *silence_duration_s* seconds is detected, or *max_record_time_s*
    is reached.

    Returns the saved WAV file path, or None if nothing was captured.
    """
    if pre_record_delay_s > 0:
        time.sleep(pre_record_delay_s)

    samplerate = _get_input_samplerate()
    chunk_duration_s = 0.05
    chunk_size = int(samplerate * chunk_duration_s)
    num_silent_chunks_to_stop = int(silence_duration_s / chunk_duration_s)
    max_chunks = int(max_record_time_s / chunk_duration_s)

    buffer: list[np.ndarray] = []
    silent_chunks = 0
    recorded_chunks = 0
    stop_flag = threading.Event()

    def _callback(indata: np.ndarray, frames: int, time_info, status):
        nonlocal silent_chunks, recorded_chunks
        volume = float(np.linalg.norm(indata) / np.sqrt(max(len(indata), 1)))
        buffer.append(indata.copy())
        recorded_chunks += 1

        # Ignore the first few chunks (mic warm-up noise)
        if recorded_chunks < 5:
            return

        if volume < silence_threshold:
            silent_chunks += 1
            if silent_chunks >= num_silent_chunks_to_stop:
                stop_flag.set()
        else:
            silent_chunks = 0

    print("[Recorder] Listening (adaptive VAD)…", flush=True)
    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            device=device,
            callback=_callback,
        ):
            while not stop_flag.is_set() and recorded_chunks < max_chunks:
                sd.sleep(int(chunk_duration_s * 1000))
    except Exception as exc:
        print(f"[Recorder] InputStream error: {exc}", flush=True)
        return None

    return _save_buffer(buffer, filename, samplerate)
