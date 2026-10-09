"""Countdown timer helpers for the frontend overlay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


def format_mmss(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remainder:02d}"


@dataclass(slots=True)
class CountdownTimer:
    """Simple countdown timer."""

    duration_s: float
    remaining_s: float = field(init=False)
    running: bool = False

    def __post_init__(self) -> None:
        self.remaining_s = max(0.0, float(self.duration_s))

    def start(self) -> None:
        self.running = True

    def update(self, dt_s: float) -> None:
        if not self.running or self.remaining_s <= 0.0:
            return
        self.remaining_s = max(0.0, self.remaining_s - max(0.0, dt_s))
        if self.remaining_s <= 0.0:
            self.running = False

    @property
    def progress(self) -> float:
        if self.duration_s <= 0.0:
            return 0.0
        return 1.0 - (self.remaining_s / self.duration_s)


@dataclass(slots=True)
class PomodoroSession:
    """Track a multi-cycle pomodoro with focus and break phases."""

    work_seconds: float = 25.0 * 60.0
    break_seconds: float = 5.0 * 60.0
    cycles: int = 4
    current_cycle: int = 1
    phase: Literal["focus", "break", "done"] = "focus"
    remaining_s: float = field(init=False)
    running: bool = False
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        self.work_seconds = max(1.0, float(self.work_seconds))
        self.break_seconds = max(0.0, float(self.break_seconds))
        self.cycles = max(1, int(self.cycles))
        self.remaining_s = self.work_seconds

    def start(self) -> None:
        self.running = True

    def _advance_phase(self, overflow_s: float) -> None:
        if self.phase == "focus":
            if self.current_cycle >= self.cycles:
                self.phase = "done"
                self.running = False
                self.remaining_s = 0.0
                return

            self.phase = "break"
            self.remaining_s = self.break_seconds - overflow_s
            if self.remaining_s <= 0.0:
                self._advance_phase(-self.remaining_s)
            return

        if self.phase == "break":
            self.current_cycle += 1
            if self.current_cycle > self.cycles:
                self.phase = "done"
                self.running = False
                self.remaining_s = 0.0
                return

            self.phase = "focus"
            self.remaining_s = self.work_seconds - overflow_s
            if self.remaining_s <= 0.0:
                self._advance_phase(-self.remaining_s)

    def update(self, dt_s: float) -> None:
        if not self.running or self.phase == "done":
            return

        self.remaining_s -= max(0.0, dt_s)
        self.elapsed_s += max(0.0, dt_s)
        while self.running and self.remaining_s <= 0.0 and self.phase != "done":
            overflow_s = -self.remaining_s
            self._advance_phase(overflow_s)

    @property
    def progress(self) -> float:
        total_s = self.total_duration_s
        if total_s <= 0.0:
            return 0.0
        return min(1.0, self.elapsed_s / total_s)

    @property
    def total_duration_s(self) -> float:
        return (self.cycles * self.work_seconds) + (max(0, self.cycles - 1) * self.break_seconds)

    @property
    def status_text(self) -> str:
        if self.phase == "done":
            return "Pomodoro complete"
        return f"{self.phase.title()} {self.current_cycle}/{self.cycles}"
