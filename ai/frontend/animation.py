"""Emotion animation loader and frame stepping logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

from ai.frontend.config import DEFAULT_FACE_CANVAS_SIZE, EMOTION_FRAME_DURATIONS_MS
from ai.frontend.state import Emotion, normalize_emotion


@dataclass(slots=True)
class AnimationClip:
    """Frames and frame timing for a single emotion."""

    frames: list[pygame.Surface]
    frame_duration_ms: int


class FaceAnimator:
    """Stateful emotion animator with per-emotion clip playback."""

    def __init__(self, clips: dict[Emotion, AnimationClip]) -> None:
        self._clips = clips
        self._emotion = Emotion.HAPPY
        self._frame_index = 0
        self._elapsed_ms = 0
        self._playing = False

    def set_emotion(self, emotion: str | Emotion) -> None:
        normalized = normalize_emotion(emotion)
        if normalized == self._emotion:
            return
        self._emotion = normalized
        self._frame_index = 0
        self._elapsed_ms = 0

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        if not self._playing:
            self._frame_index = 0
            self._elapsed_ms = 0

    def update(self, dt_ms: int) -> None:
        if not self._playing:
            return
        clip = self._clips[self._emotion]
        if len(clip.frames) <= 1:
            return
        self._elapsed_ms += max(0, int(dt_ms))
        while self._elapsed_ms >= clip.frame_duration_ms:
            self._elapsed_ms -= clip.frame_duration_ms
            self._frame_index = (self._frame_index + 1) % len(clip.frames)

    def current_frame(self) -> pygame.Surface:
        clip = self._clips[self._emotion]
        if not self._playing:
            return clip.frames[0]
        return clip.frames[self._frame_index % len(clip.frames)]


def load_face_clips(asset_root: Path, canvas_size: tuple[int, int] = DEFAULT_FACE_CANVAS_SIZE) -> dict[Emotion, AnimationClip]:
    """Load the PNG frame sequence for every emotion from <asset_root>/faces/<emotion>/."""

    clips: dict[Emotion, AnimationClip] = {}
    for emotion in Emotion:
        emotion_dir = asset_root / "faces" / emotion.value
        png_paths = sorted(emotion_dir.glob("*.png"))
        if not png_paths:
            raise FileNotFoundError(f"No face frames found in {emotion_dir}")
        frames = [
            pygame.transform.smoothscale(pygame.image.load(str(path)).convert_alpha(), canvas_size)
            for path in png_paths
        ]
        frame_duration_ms = EMOTION_FRAME_DURATIONS_MS.get(emotion.value, 180)
        clips[emotion] = AnimationClip(frames=frames, frame_duration_ms=frame_duration_ms)
    return clips
