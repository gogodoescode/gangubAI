"""Wake word detector ("Hi Gungu Bai") using OpenWakeWord."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal
import sounddevice as sd
from openwakeword.model import Model

from ai.voice import config


class WakeWordDetector:
    """Blocks on the microphone until the wake word is heard."""

    def __init__(
        self,
        wake_word_model: str = config.WAKE_WORD_MODEL,
        feature_models_dir: str = config.WAKE_WORD_ASSETS_DIR,
        threshold: float = config.WAKE_WORD_THRESHOLD,
        input_device: str | int | None = config.INPUT_DEVICE_NAME,
    ) -> None:
        model_path = Path(wake_word_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Wake word model not found: {model_path}. Set ai.voice.config.WAKE_WORD_MODEL."
            )

        # Use local feature models when present, otherwise OpenWakeWord's bundled ones.
        model_kwargs: dict[str, str] = {}
        assets_dir = Path(feature_models_dir)
        melspec_path = assets_dir / "melspectrogram.onnx"
        embedding_path = assets_dir / "embedding_model.onnx"
        if melspec_path.exists() and embedding_path.exists():
            model_kwargs["melspec_model_path"] = str(melspec_path)
            model_kwargs["embedding_model_path"] = str(embedding_path)

        self._model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            **model_kwargs,
        )
        self.threshold = threshold
        self.input_device = input_device
        print(f"[WakeWord] Loaded wake-word model: {model_path}", flush=True)

    def _input_samplerate(self) -> int:
        try:
            info = sd.query_devices(device=self.input_device, kind="input")
            return int(info["default_samplerate"])
        except Exception:
            return 48000

    def wait_for_wake_word(self) -> None:
        """Listen on the microphone and return once the wake word is detected."""
        self._model.reset()

        # Record at the mic's native rate and resample each chunk to 16 kHz.
        native_rate = self._input_samplerate()
        input_chunk = int(config.OWW_CHUNK_SIZE * native_rate / config.OWW_SAMPLE_RATE)

        print("[WakeWord] Listening for wake word...", flush=True)
        with sd.InputStream(
            samplerate=native_rate,
            channels=1,
            dtype="int16",
            blocksize=input_chunk,
            device=self.input_device,
        ) as stream:
            while True:
                data, _ = stream.read(input_chunk)
                audio = data[:, 0]
                if native_rate != config.OWW_SAMPLE_RATE:
                    audio = scipy.signal.resample(audio, config.OWW_CHUNK_SIZE).astype(np.int16)

                scores = self._model.predict(audio)
                if max(scores.values(), default=0.0) >= self.threshold:
                    self._model.reset()
                    return
