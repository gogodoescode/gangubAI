"""Configuration for the GangubAI Pygame frontend."""

from __future__ import annotations

from pathlib import Path

ASSETS_ROOT = Path(__file__).resolve().parent / "assets"

DEFAULT_WINDOW_SIZE: tuple[int, int] = (800, 480)
DEFAULT_FACE_CANVAS_SIZE: tuple[int, int] = (420, 320)
FRONTEND_FPS = 60

# Localhost UDP port the voice app sends emotion events to.
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 8765

EMOTION_FRAME_DURATIONS_MS: dict[str, int] = {
    "neutral": 180,
    "happy": 140,
    "sad": 220,
    "angry": 160,
    "curious": 180,
    "excited": 110,
    "confused": 180,
    "thinking": 150,
}
