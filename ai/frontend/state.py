"""Frontend state and emotion primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time


class Emotion(str, Enum):
    """Emotion labels shared by the robot UI."""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CURIOUS = "curious"
    EXCITED = "excited"
    CONFUSED = "confused"
    NEUTRAL = "neutral"
    THINKING = "thinking"


_VALID_EMOTIONS = {emotion.value for emotion in Emotion}


def normalize_emotion(value: str | Emotion | None) -> Emotion:
    """Map arbitrary input to a known emotion label."""

    if isinstance(value, Emotion):
        return value

    if value is None:
        return Emotion.HAPPY

    normalized = str(value).strip().lower()
    if normalized in _VALID_EMOTIONS:
        return Emotion(normalized)
    return Emotion.NEUTRAL


@dataclass(slots=True)
class EmotionEvent:
    """Emotion update received from an external source."""

    emotion: Emotion
    source: str = "unknown"
    timestamp: float = field(default_factory=time)
    payload: str = ""


@dataclass(slots=True)
class FrontendState:
    """Mutable runtime state for the Pygame frontend."""

    emotion: Emotion = Emotion.HAPPY
    source: str = "demo"
    last_update_s: float = field(default_factory=time)
    last_payload: str = ""
    speaking: bool = False
    running: bool = True
    fullscreen: bool = False
