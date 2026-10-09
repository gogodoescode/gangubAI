"""Overlay rendering for the GangubAI frontend."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from ai.frontend.timers import CountdownTimer, PomodoroSession, format_mmss


@dataclass(slots=True)
class FontPack:
    title: pygame.font.Font
    body: pygame.font.Font


def load_fonts() -> FontPack:
    pygame.font.init()
    font_name = pygame.font.match_font("dejavusans") or pygame.font.get_default_font()
    title = pygame.font.Font(font_name, 34)
    body = pygame.font.Font(font_name, 22)
    return FontPack(title=title, body=body)


def draw_overlay_card(
    surface: pygame.Surface,
    fonts: FontPack,
    title: str,
    subtitle_lines: list[str],
    progress: float,
    accent: tuple[int, int, int],
) -> None:
    width = min(720, surface.get_width() - 80)
    height = 176 if len(subtitle_lines) <= 2 else 200
    x = (surface.get_width() - width) // 2
    y = surface.get_height() - height - 28

    card = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(card, (8, 10, 16, 188), card.get_rect(), border_radius=24)
    pygame.draw.rect(card, (*accent, 220), card.get_rect(), width=2, border_radius=24)

    card.blit(fonts.title.render(title, True, (248, 248, 252)), (22, 18))
    text_y = 62
    for line in subtitle_lines:
        card.blit(fonts.body.render(line, True, (224, 230, 240)), (24, text_y))
        text_y += 26

    bar_rect = pygame.Rect(24, height - 28, width - 48, 10)
    pygame.draw.rect(card, (255, 255, 255, 28), bar_rect, border_radius=6)
    progress_width = int(bar_rect.width * max(0.0, min(1.0, progress)))
    if progress_width > 0:
        pygame.draw.rect(card, (*accent, 230), (bar_rect.x, bar_rect.y, progress_width, bar_rect.height), border_radius=6)

    surface.blit(card, (x, y))


def draw_timer_overlay(surface: pygame.Surface, fonts: FontPack, timer: CountdownTimer, accent: tuple[int, int, int]) -> None:
    draw_overlay_card(
        surface=surface,
        fonts=fonts,
        title="Timer",
        subtitle_lines=[f"Remaining {format_mmss(timer.remaining_s)}"],
        progress=timer.progress,
        accent=accent,
    )


def draw_pomodoro_overlay(surface: pygame.Surface, fonts: FontPack, session: PomodoroSession, accent: tuple[int, int, int]) -> None:
    phase_line = f"{session.phase.title()} {format_mmss(session.remaining_s)}"
    cycle_line = session.status_text
    draw_overlay_card(
        surface=surface,
        fonts=fonts,
        title="Pomodoro",
        subtitle_lines=[phase_line, cycle_line],
        progress=session.progress,
        accent=accent,
    )

