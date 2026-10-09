"""Runnable Pygame frontend for the GangubAI robot face."""

from __future__ import annotations

import argparse
import json
import os
import time

import pygame

from ai.frontend.animation import FaceAnimator, load_face_clips
from ai.frontend.bridge.local_queue import EmotionListener
from ai.frontend.config import (
    ASSETS_ROOT,
    DEFAULT_FACE_CANVAS_SIZE,
    DEFAULT_WINDOW_SIZE,
    FRONTEND_FPS,
)
from ai.frontend.overlays import draw_pomodoro_overlay, draw_timer_overlay, load_fonts
from ai.frontend.state import Emotion, FrontendState, normalize_emotion
from ai.frontend.timers import CountdownTimer, PomodoroSession


# Disable Pygame's audio driver to avoid exclusive ALSA device access.
# This allows the backend TTS (aplay) to use the audio device simultaneously.
os.environ['SDL_AUDIODRIVER'] = 'dummy'


KEY_TO_EMOTION: dict[int, Emotion] = {
    pygame.K_1: Emotion.HAPPY,
    pygame.K_2: Emotion.HAPPY,
    pygame.K_3: Emotion.SAD,
    pygame.K_4: Emotion.ANGRY,
    pygame.K_5: Emotion.CURIOUS,
    pygame.K_6: Emotion.EXCITED,
    pygame.K_7: Emotion.CONFUSED,
    pygame.K_8: Emotion.THINKING,
}


class FrontendApp:
    """Owns the Pygame window, animation state, and event loop."""

    def __init__(self, fullscreen: bool = False) -> None:
        pygame.init()
        pygame.display.set_caption("GangubAI Face")
        pygame.key.set_repeat(250, 40)

        self.fonts = load_fonts()
        self.state = FrontendState(fullscreen=fullscreen)
        self.clock = pygame.time.Clock()
        self.windowed_size = DEFAULT_WINDOW_SIZE
        self.screen = self._create_display(fullscreen)
        self.canvas_size = DEFAULT_FACE_CANVAS_SIZE
        self.animator = FaceAnimator(load_face_clips(ASSETS_ROOT, self.canvas_size))
        self.animator.set_emotion(Emotion.HAPPY)
        self.animator.set_playing(False)
        self.listener = EmotionListener()
        self.timer_overlay: CountdownTimer | None = None
        self.pomodoro_overlay: PomodoroSession | None = None

    def _create_display(self, fullscreen: bool) -> pygame.Surface:
        flags = pygame.FULLSCREEN if fullscreen else 0
        size = pygame.display.get_desktop_sizes()[0] if fullscreen else self.windowed_size
        return pygame.display.set_mode(size, flags)

    def _toggle_fullscreen(self) -> None:
        self.state.fullscreen = not self.state.fullscreen
        self.screen = self._create_display(self.state.fullscreen)

    def _apply_emotion(self, emotion: str | Emotion, source: str, payload: str = "") -> None:
        normalized = normalize_emotion(emotion)
        self.state.emotion = normalized
        self.state.source = source
        self.state.last_update_s = time.time()
        self.state.last_payload = payload
        self.animator.set_emotion(normalized)

    def _set_idle_happy(self, source: str = "idle") -> None:
        self.state.speaking = False
        self._apply_emotion(Emotion.HAPPY, source=source)
        self.animator.set_playing(False)

    def _handle_bridge_event(self, emotion: Emotion, source: str, payload: str) -> None:
        event_type = "emotion"
        event_data: dict[str, object] = {}
        if payload:
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, dict):
                    event_type = str(decoded.get("event", "emotion"))
                    raw_payload = decoded.get("payload", {})
                    if isinstance(raw_payload, dict):
                        event_data = raw_payload
            except Exception:
                event_type = payload.strip() or "emotion"

        if event_type == "timer_start":
            duration_seconds = float(event_data.get("duration_seconds", 0.0) or 0.0)
            if duration_seconds > 0.0:
                self.timer_overlay = CountdownTimer(duration_seconds)
                self.timer_overlay.start()
                self.pomodoro_overlay = None
            return

        if event_type == "pomodoro_start":
            work_seconds = float(event_data.get("work_seconds", 25.0 * 60.0) or 25.0 * 60.0)
            break_seconds = float(event_data.get("break_seconds", 5.0 * 60.0) or 5.0 * 60.0)
            cycles = int(event_data.get("cycles", 4) or 4)
            self.pomodoro_overlay = PomodoroSession(
                work_seconds=work_seconds,
                break_seconds=break_seconds,
                cycles=cycles,
            )
            self.pomodoro_overlay.start()
            self.timer_overlay = None
            return

        if event_type == "speech_start":
            self.state.speaking = True
            self._apply_emotion(emotion, source=source, payload=payload)
            self.animator.set_playing(True)
            return

        if event_type == "speech_end":
            self._set_idle_happy(source=source)
            return

        # While speaking, allow emotion updates; otherwise stay idle happy.
        if self.state.speaking:
            self._apply_emotion(emotion, source=source, payload=payload)

    def _handle_key(self, event: pygame.event.Event) -> None:
        if event.key in KEY_TO_EMOTION:
            self._apply_emotion(KEY_TO_EMOTION[event.key], source="keyboard")
            return

        if event.key == pygame.K_f:
            self._toggle_fullscreen()
            return

        if event.key in {pygame.K_ESCAPE, pygame.K_q}:
            self.state.running = False

    def _drain_listener(self) -> None:
        event = self.listener.poll_latest()
        if event is None:
            return
        if event.timestamp and (time.time() - event.timestamp) > 60.0:
            return
        self._handle_bridge_event(event.emotion, source=event.source, payload=event.payload)

    def run(self) -> int:
        self._set_idle_happy(source="startup")
        while self.state.running:
            dt_ms = self.clock.tick(FRONTEND_FPS)
            dt_s = dt_ms / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event)

            self._drain_listener()
            self.animator.update(dt_ms)

            if self.timer_overlay is not None:
                self.timer_overlay.update(dt_s)
                if self.timer_overlay.remaining_s <= 0.0:
                    self.timer_overlay = None

            if self.pomodoro_overlay is not None:
                self.pomodoro_overlay.update(dt_s)
                if self.pomodoro_overlay.phase == "done":
                    self.pomodoro_overlay = None

            self.screen.fill((0, 0, 0))
            self._draw_face()
            self._draw_overlay()
            pygame.display.flip()

        self.shutdown()
        return 0

    def _draw_face(self) -> None:
        frame = self.animator.current_frame()
        screen_rect = self.screen.get_rect()
        scaled = pygame.transform.smoothscale(frame, screen_rect.size)
        self.screen.blit(scaled, (0, 0))

    def _draw_overlay(self) -> None:
        accent = (106, 199, 255)
        if self.timer_overlay is not None:
            draw_timer_overlay(self.screen, self.fonts, self.timer_overlay, accent)
            return

        if self.pomodoro_overlay is not None:
            draw_pomodoro_overlay(self.screen, self.fonts, self.pomodoro_overlay, accent)

    def shutdown(self) -> None:
        self.listener.close()
        pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description="GangubAI Pygame frontend")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = parser.parse_args()
    return FrontendApp(fullscreen=args.fullscreen).run()


if __name__ == "__main__":
    raise SystemExit(main())
