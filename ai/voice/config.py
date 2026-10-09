"""Audio configuration for the GangubAI voice pipeline."""

# ── Recording ──────────────────────────────────────────────
# Microphone device name or index. None = system default.
INPUT_DEVICE_NAME: str | None = None

# RMS amplitude below this value is treated as silence.
SILENCE_THRESHOLD: float = 0.006

# How long continuous silence must last before recording stops (seconds).
# Reduced from 5.0 for faster response; adjust down for more sensitive stopping.
SILENCE_DURATION_S: float = 1.5

# Hard cap on recording length (seconds).
MAX_RECORD_TIME_S: float = 30.0

# Seconds to wait after trigger before starting to record
# (lets the wake-word sound die out so it isn't captured).
PRE_RECORD_DELAY_S: float = 0.4

# TTS output backend:
#   True  -> ALSA `aplay` with S32_LE stereo (Raspberry Pi + MAX98357A I2S amp)
#   False -> sounddevice on OUTPUT_DEVICE_NAME (laptop / PC speakers)
TTS_USE_APLAY: bool = True

# Temporary file used for the recorded audio.
RECORDING_OUTPUT_FILE: str = "gangubai_input.wav"


# -- Wake word (OpenWakeWord) ---------------------------------------------
# Path to wake word model (.onnx).
WAKE_WORD_MODEL: str = "./wakeword_models/hi_gungu_bai.onnx"

# Directory containing OpenWakeWord feature models
# (melspectrogram.onnx, embedding_model.onnx, silero_vad.onnx).
WAKE_WORD_ASSETS_DIR: str = "./wakeword_models"

# Trigger threshold for wake word confidence.
WAKE_WORD_THRESHOLD: float = 0.5

# OpenWakeWord required sample rate.
OWW_SAMPLE_RATE: int = 16000

# Frame size fed to OpenWakeWord predict().
OWW_CHUNK_SIZE: int = 1280


# -- Transcription (whisper.cpp only) -------------------------------------
# Path to whisper.cpp CLI binary (relative to repo root by default).
WHISPER_CPP_BIN: str = "./whisper.cpp/build/bin/whisper-cli"

# Path to whisper.cpp model file.
WHISPER_CPP_MODEL: str = "./whisper.cpp/models/ggml-base.en.bin"

# Transcription language passed to whisper.cpp.
WHISPER_LANGUAGE: str = "en"

# Number of CPU threads used by whisper.cpp.
WHISPER_CPP_THREADS: int = 4

# Hard timeout for a transcription subprocess call.
WHISPER_TIMEOUT_S: int = 90


# -- TTS (Piper) -----------------------------------------------------------
# Path to Piper executable.
PIPER_BIN: str = "./piper/piper/piper"

# Path to Piper voice model (.onnx).
PIPER_VOICE_MODEL: str = "./piper/en_GB-semaine-medium.onnx"

# Piper raw PCM sample rate.
PIPER_SAMPLE_RATE: int = 22050

# sounddevice output device name or index (TTS_USE_APLAY=False only). None = system default.
OUTPUT_DEVICE_NAME: str | int | None = None

# Raw bytes read from Piper stdout per loop iteration.
# Increased from 4096 to 32KB to buffer ~1.45s of audio (reduces ALSA underruns).
TTS_READ_CHUNK_BYTES: int = 32768

# Output stream block size used by sounddevice.
TTS_STREAM_BLOCKSIZE: int = 8192

# Queue chunk size for TTS text. Larger chunks reduce sentence-to-sentence gaps.
TTS_QUEUE_CHUNK_CHARS: int = 700

# Small silence tail (seconds) appended to each spoken chunk to avoid clipped endings.
TTS_TAIL_SILENCE_S: float = 0.01
