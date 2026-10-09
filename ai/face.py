"""GangubAI face: animated emotion display with timer / pomodoro overlays.

Other processes talk to the face with `send_to_face()` over localhost UDP.

Usage:
    python3 -m ai.face [--fullscreen]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# Disable Pygame's audio driver so the TTS (aplay) can use the sound device.
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame  # noqa: E402

from ai import config  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# Emotions + UDP link
# ═══════════════════════════════════════════════════════════════════

class Emotion(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CURIOUS = "curious"
    EXCITED = "excited"
    CONFUSED = "confused"
    NEUTRAL = "neutral"
    THINKING = "thinking"


def normalize_emotion(value: str | Emotion | None) -> Emotion:
    """Map arbitrary input to a known emotion (unknown -> neutral)."""
    if isinstance(value, Emotion):
        return value
    try:
        return Emotion(str(value or "happy").strip().lower())
    except ValueError:
        return Emotion.NEUTRAL


_send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_to_face(emotion: str, event: str = "emotion", payload: dict | None = None) -> None:
    """Best-effort: tell the face which emotion to show and what happened.

    event: "speech_start" | "speech_end" | "timer_start" | "pomodoro_start" | "emotion"
    """
    packet = {
        "emotion": normalize_emotion(emotion).value,
        "event": event,
        "payload": payload or {},
        "timestamp": time.time(),
    }
    try:
        _send_socket.sendto(json.dumps(packet).encode("utf-8"), (config.FACE_HOST, config.FACE_PORT))
    except OSError:
        pass


class FaceListener:
    """Non-blocking UDP listener; returns only the newest packet."""

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((config.FACE_HOST, config.FACE_PORT))
        self._socket.setblocking(False)

    def poll_latest(self) -> dict | None:
        latest = None
        while True:
            try:
                raw, _ = self._socket.recvfrom(8192)
            except OSError:
                return latest
            try:
                packet = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            if isinstance(packet, dict):
                latest = packet

    def close(self) -> None:
        self._socket.close()


# ═══════════════════════════════════════════════════════════════════
# Timers
# ═══════════════════════════════════════════════════════════════════

def format_mmss(seconds: float) -> str:
    minutes, remainder = divmod(max(0, int(round(seconds))), 60)
    return f"{minutes:02d}:{remainder:02d}"


@dataclass(slots=True)
class CountdownTimer:
    duration_s: float
    remaining_s: float = field(init=False)

    def __post_init__(self) -> None:
        self.remaining_s = max(0.0, float(self.duration_s))

    def update(self, dt_s: float) -> None:
        self.remaining_s = max(0.0, self.remaining_s - max(0.0, dt_s))

    @property
    def done(self) -> bool:
        return self.remaining_s <= 0.0

    @property
    def progress(self) -> float:
        return 1.0 - (self.remaining_s / self.duration_s) if self.duration_s > 0 else 0.0


@dataclass(slots=True)
class PomodoroSession:
    """Multi-cycle pomodoro with focus and break phases."""

    work_seconds: float = 25.0 * 60.0
    break_seconds: float = 5.0 * 60.0
    cycles: int = 4
    current_cycle: int = 1
    phase: Literal["focus", "break", "done"] = "focus"
    remaining_s: float = field(init=False)
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        self.work_seconds = max(1.0, float(self.work_seconds))
        self.break_seconds = max(0.0, float(self.break_seconds))
        self.cycles = max(1, int(self.cycles))
        self.remaining_s = self.work_seconds

    def update(self, dt_s: float) -> None:
        if self.done:
            return
        self.remaining_s -= max(0.0, dt_s)
        self.elapsed_s += max(0.0, dt_s)
        while self.remaining_s <= 0.0 and not self.done:
            overflow_s = -self.remaining_s
            if self.phase == "focus" and self.current_cycle < self.cycles:
                self.phase, self.remaining_s = "break", self.break_seconds - overflow_s
            elif self.phase == "break":
                self.current_cycle += 1
                self.phase, self.remaining_s = "focus", self.work_seconds - overflow_s
            else:
                self.phase, self.remaining_s = "done", 0.0

    @property
    def done(self) -> bool:
        return self.phase == "done"

    @property
    def progress(self) -> float:
        total_s = self.cycles * self.work_seconds + (self.cycles - 1) * self.break_seconds
        return min(1.0, self.elapsed_s / total_s)

    @property
    def status_text(self) -> str:
        if self.done:
            return "Pomodoro complete"
        return f"{self.phase.title()} {self.current_cycle}/{self.cycles}"


# ═══════════════════════════════════════════════════════════════════
# Drawing
# ═══════════════════════════════════════════════════════════════════

OVERLAY_ACCENT = (106, 199, 255)


def draw_overlay_card(surface: pygame.Surface, fonts: tuple[pygame.font.Font, pygame.font.Font],
                      title: str, lines: list[str], progress: float) -> None:
    """Rounded card at the bottom of the screen with a title, text lines and a progress bar."""
    title_font, body_font = fonts
    width = min(720, surface.get_width() - 80)
    height = 176
    card = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(card, (8, 10, 16, 188), card.get_rect(), border_radius=24)
    pygame.draw.rect(card, (*OVERLAY_ACCENT, 220), card.get_rect(), width=2, border_radius=24)

    card.blit(title_font.render(title, True, (248, 248, 252)), (22, 18))
    for i, line in enumerate(lines):
        card.blit(body_font.render(line, True, (224, 230, 240)), (24, 62 + i * 26))

    bar = pygame.Rect(24, height - 28, width - 48, 10)
    pygame.draw.rect(card, (255, 255, 255, 28), bar, border_radius=6)
    filled = int(bar.width * max(0.0, min(1.0, progress)))
    if filled > 0:
        pygame.draw.rect(card, (*OVERLAY_ACCENT, 230), (bar.x, bar.y, filled, bar.height), border_radius=6)

    surface.blit(card, ((surface.get_width() - width) // 2, surface.get_height() - height - 28))


class FaceAnimator:
    """Plays the PNG frame loop for the current emotion while speaking."""

    def __init__(self) -> None:
        self._clips: dict[Emotion, list[pygame.Surface]] = {}
        for emotion in Emotion:
            paths = sorted((config.FACES_DIR / emotion.value).glob("*.png"))
            if not paths:
                raise FileNotFoundError(f"No face frames found in {config.FACES_DIR / emotion.value}")
            self._clips[emotion] = [
                pygame.transform.smoothscale(pygame.image.load(str(p)).convert_alpha(), config.FACE_CANVAS_SIZE)
                for p in paths
            ]
        self.emotion = Emotion.HAPPY
        self.playing = False
        self._frame_index = 0
        self._elapsed_ms = 0

    def set_emotion(self, emotion: Emotion) -> None:
        if emotion != self.emotion:
            self.emotion = emotion
            self._frame_index = self._elapsed_ms = 0

    def set_playing(self, playing: bool) -> None:
        self.playing = playing
        if not playing:
            self._frame_index = self._elapsed_ms = 0

    def update(self, dt_ms: int) -> None:
        if not self.playing:
            return
        frame_ms = config.EMOTION_FRAME_DURATIONS_MS.get(self.emotion.value, 180)
        self._elapsed_ms += dt_ms
        while self._elapsed_ms >= frame_ms:
            self._elapsed_ms -= frame_ms
            self._frame_index = (self._frame_index + 1) % len(self._clips[self.emotion])

    def current_frame(self) -> pygame.Surface:
        return self._clips[self.emotion][self._frame_index if self.playing else 0]


# ═══════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════

KEY_TO_EMOTION = {
    pygame.K_1: Emotion.HAPPY,
    pygame.K_2: Emotion.HAPPY,
    pygame.K_3: Emotion.SAD,
    pygame.K_4: Emotion.ANGRY,
    pygame.K_5: Emotion.CURIOUS,
    pygame.K_6: Emotion.EXCITED,
    pygame.K_7: Emotion.CONFUSED,
    pygame.K_8: Emotion.THINKING,
}


class FaceApp:
    """Owns the Pygame window and event loop."""

    def __init__(self, fullscreen: bool = False) -> None:
        pygame.init()
        pygame.display.set_caption("GangubAI Face")
        font_name = pygame.font.match_font("dejavusans") or pygame.font.get_default_font()
        self.fonts = (pygame.font.Font(font_name, 34), pygame.font.Font(font_name, 22))
        self.fullscreen = fullscreen
        self.screen = self._create_display()
        self.clock = pygame.time.Clock()
        self.animator = FaceAnimator()
        self.listener = FaceListener()
        self.speaking = False
        self.running = True
        self.overlay: CountdownTimer | PomodoroSession | None = None

    def _create_display(self) -> pygame.Surface:
        if self.fullscreen:
            return pygame.display.set_mode(pygame.display.get_desktop_sizes()[0], pygame.FULLSCREEN)
        return pygame.display.set_mode(config.WINDOW_SIZE)

    def _handle_packet(self, packet: dict) -> None:
        sent_at = float(packet.get("timestamp") or 0)
        if sent_at and time.time() - sent_at > 60.0:
            return  # stale
        emotion = normalize_emotion(packet.get("emotion"))
        event = packet.get("event", "emotion")
        data = packet.get("payload") or {}

        if event == "timer_start":
            duration_s = float(data.get("duration_seconds", 0) or 0)
            if duration_s > 0:
                self.overlay = CountdownTimer(duration_s)
        elif event == "pomodoro_start":
            self.overlay = PomodoroSession(
                work_seconds=float(data.get("work_seconds", 25 * 60)),
                break_seconds=float(data.get("break_seconds", 5 * 60)),
                cycles=int(data.get("cycles", 4)),
            )
        elif event == "speech_start":
            self.speaking = True
            self.animator.set_emotion(emotion)
            self.animator.set_playing(True)
        elif event == "speech_end":
            self.speaking = False
            self.animator.set_emotion(Emotion.HAPPY)
            self.animator.set_playing(False)
        elif self.speaking:
            # Outside speech the face stays idle happy.
            self.animator.set_emotion(emotion)

    def _handle_key(self, key: int) -> None:
        if key in KEY_TO_EMOTION:
            self.animator.set_emotion(KEY_TO_EMOTION[key])
        elif key == pygame.K_f:
            self.fullscreen = not self.fullscreen
            self.screen = self._create_display()
        elif key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False

    def run(self) -> int:
        try:
            while self.running:
                dt_ms = self.clock.tick(config.FPS)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        self._handle_key(event.key)

                packet = self.listener.poll_latest()
                if packet is not None:
                    self._handle_packet(packet)
                self.animator.update(dt_ms)

                self.screen.blit(pygame.transform.smoothscale(self.animator.current_frame(), self.screen.get_size()), (0, 0))
                if self.overlay is not None:
                    self.overlay.update(dt_ms / 1000.0)
                    if self.overlay.done:
                        self.overlay = None
                    elif isinstance(self.overlay, CountdownTimer):
                        draw_overlay_card(self.screen, self.fonts, "Timer",
                                          [f"Remaining {format_mmss(self.overlay.remaining_s)}"], self.overlay.progress)
                    else:
                        draw_overlay_card(self.screen, self.fonts, "Pomodoro",
                                          [f"{self.overlay.phase.title()} {format_mmss(self.overlay.remaining_s)}",
                                           self.overlay.status_text], self.overlay.progress)
                pygame.display.flip()
        finally:
            self.listener.close()
            pygame.quit()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="GangubAI face")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    return FaceApp(fullscreen=parser.parse_args().fullscreen).run()


if __name__ == "__main__":
    raise SystemExit(main())
