"""Reusable, telemetry-only Pygame UI for observing a Pacman DQN agent.

The renderer deliberately owns no learning state.  Every number and every
neural-network edge comes from the telemetry passed to :meth:`render`, making
it safe to reuse with a live agent, a recorded run, or an empty initial state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any, Optional

import pygame

from .theme import DEFAULT_THEME, Color, ObservatoryTheme


class ObservatoryTab(str, Enum):
    """The four top-level views exposed by the observatory."""

    GAME = "GAME"
    VISION = "VISION"
    METRICS = "METRICS"
    NETWORK = "NETWORK"


@dataclass(frozen=True)
class ObservatoryLayout:
    """Geometry from the latest render, useful for host-side event routing."""

    header: pygame.Rect
    content: pygame.Rect
    tabs: Mapping[ObservatoryTab, pygame.Rect]
    speed_presets: tuple[pygame.Rect, ...] = ()


@dataclass(frozen=True)
class _Layer:
    name: str
    size: Optional[int]
    activations: tuple[Optional[float], ...]
    full_size: Optional[int] = None


_MISSING = object()
_FONT_PATH = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "arial.ttf"


class PacmanObservatory:
    """Draw a tabbed DQN observability UI into any Pygame surface.

    The primary API is::

        ui = PacmanObservatory()
        ui.handle_event(event)  # optional; F1-F4, Tab, or tab clicks
        ui.render(window, telemetry, history=history, game_surface=game)

    ``telemetry`` is a mapping.  Common aliases are accepted for scalar
    metrics and Q-values.  A neural-network payload can be provided under
    ``telemetry["network"]`` with ``layers``, ``activations``, and ``weights``.
    Weight matrices use the PyTorch convention ``[output][input]`` by default;
    set ``weight_layout="in_out"`` when the matrices use the opposite layout.
    Missing fields are rendered as explicit empty states, never as synthetic
    zeroes or fabricated history.

    When speed telemetry is supplied, the header exposes preset hitboxes in
    :class:`ObservatoryLayout`; hosts can route a click with
    :meth:`speed_preset_at` while retaining ownership of the runtime rate.
    """

    MIN_WIDTH = 620
    # The metrics view needs enough room for health, charts, and replay. Below
    # this threshold an explicit resize state is safer than clipped evidence.
    MIN_HEIGHT = 580
    HEADER_HEIGHT = 76
    PADDING = 16

    def __init__(
        self,
        *,
        initial_tab: ObservatoryTab | str = ObservatoryTab.GAME,
        theme: ObservatoryTheme = DEFAULT_THEME,
        font_path: str | Path | None = None,
        chart_window: int = 120,
        max_visible_neurons: int = 11,
    ) -> None:
        pygame.font.init()
        self.theme = theme
        self.active_tab = self._coerce_tab(initial_tab)
        self.chart_window = max(8, int(chart_window))
        self.max_visible_neurons = max(3, int(max_visible_neurons))
        self.font_path = Path(font_path) if font_path is not None else _FONT_PATH
        self._fonts: dict[tuple[int, bool], pygame.font.Font] = {}
        self._layout: Optional[ObservatoryLayout] = None

    @property
    def layout(self) -> Optional[ObservatoryLayout]:
        """Return geometry from the latest call to :meth:`render`."""

        return self._layout

    def set_tab(self, tab: ObservatoryTab | str) -> None:
        """Select a tab by enum or case-insensitive name."""

        self.active_tab = self._coerce_tab(tab)

    def speed_preset_at(self, position: tuple[int, int]) -> Optional[int]:
        """Return the clicked speed-preset index from the latest layout."""

        if self._layout is None:
            return None
        for index, rect in enumerate(self._layout.speed_presets):
            if rect.collidepoint(position):
                return index
        return None

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle observatory navigation and report whether it was consumed."""

        if event.type == pygame.KEYDOWN:
            key_tabs = {
                pygame.K_F1: ObservatoryTab.GAME,
                pygame.K_F2: ObservatoryTab.VISION,
                pygame.K_F3: ObservatoryTab.METRICS,
                pygame.K_F4: ObservatoryTab.NETWORK,
            }
            if event.key in key_tabs:
                self.active_tab = key_tabs[event.key]
                return True
            if event.key == pygame.K_TAB:
                tabs = tuple(ObservatoryTab)
                self.active_tab = tabs[(tabs.index(self.active_tab) + 1) % len(tabs)]
                return True
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and getattr(event, "button", None) == 1
            and self._layout is not None
        ):
            for tab, rect in self._layout.tabs.items():
                if rect.collidepoint(event.pos):
                    self.active_tab = tab
                    return True
        return False

    def render(
        self,
        surface: pygame.Surface,
        telemetry: Optional[Mapping[str, Any]] = None,
        *,
        history: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]] = None,
        game_surface: Optional[pygame.Surface] = None,
    ) -> ObservatoryLayout:
        """Render the selected tab and return its calculated layout.

        ``history`` may be a mapping of metric names to sequences or a sequence
        of transition/episode mappings.  When omitted, a ``history`` mapping in
        telemetry is used if present.  ``game_surface`` is copied and scaled;
        the caller retains ownership of it.
        """

        if not isinstance(surface, pygame.Surface):
            raise TypeError("surface must be a pygame.Surface")
        data: Mapping[str, Any] = telemetry if isinstance(telemetry, Mapping) else {}
        supplied_history = history
        if supplied_history is None:
            embedded = data.get("history")
            if isinstance(embedded, Mapping) or _is_record_sequence(embedded):
                supplied_history = embedded

        surface.fill(self.theme.background)
        bounds = surface.get_rect()
        header_height = min(self.HEADER_HEIGHT, max(58, bounds.height // 7))
        header = pygame.Rect(0, 0, bounds.width, header_height)
        content = pygame.Rect(
            self.PADDING,
            header.bottom + self.PADDING,
            max(0, bounds.width - self.PADDING * 2),
            max(0, bounds.height - header.bottom - self.PADDING * 2),
        )
        tab_rects, speed_rects = self._draw_header(surface, header, data)
        self._layout = ObservatoryLayout(header, content, tab_rects, speed_rects)

        if bounds.width < self.MIN_WIDTH or bounds.height < self.MIN_HEIGHT:
            self._empty_state(
                surface,
                content,
                "Surface too small",
                f"Use at least {self.MIN_WIDTH} × {self.MIN_HEIGHT} pixels",
            )
            return self._layout

        if self.active_tab is ObservatoryTab.GAME:
            self._draw_game_tab(surface, content, data, supplied_history, game_surface)
        elif self.active_tab is ObservatoryTab.VISION:
            self._draw_vision_tab(surface, content, data)
        elif self.active_tab is ObservatoryTab.METRICS:
            self._draw_metrics_tab(surface, content, data, supplied_history)
        else:
            self._draw_network_tab(surface, content, data)
        return self._layout

    @staticmethod
    def _coerce_tab(tab: ObservatoryTab | str) -> ObservatoryTab:
        if isinstance(tab, ObservatoryTab):
            return tab
        try:
            return ObservatoryTab(str(tab).strip().upper())
        except ValueError as error:
            choices = ", ".join(item.value for item in ObservatoryTab)
            raise ValueError(f"unknown tab {tab!r}; expected one of {choices}") from error

    def _font(self, size: int, *, bold: bool = False) -> pygame.font.Font:
        key = (max(8, int(size)), bold)
        if key not in self._fonts:
            try:
                font = pygame.font.Font(str(self.font_path), key[0])
            except (FileNotFoundError, OSError, pygame.error):
                font = pygame.font.Font(None, key[0])
            font.set_bold(bold)
            self._fonts[key] = font
        return self._fonts[key]

    def _draw_header(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
    ) -> tuple[dict[ObservatoryTab, pygame.Rect], tuple[pygame.Rect, ...]]:
        pygame.draw.rect(surface, self.theme.header, rect)
        pygame.draw.line(surface, self.theme.grid, rect.bottomleft, rect.bottomright, 1)

        compact = rect.width < 980
        title = "PACMAN / DQN" if compact else "PACMAN / DQN OBSERVATORY"
        self._text(surface, title, (self.PADDING, 13), self.theme.text, 17, bold=True)
        algorithm = _first(telemetry, "algorithm", "agent", "mode")
        if algorithm is not _MISSING and algorithm is not None:
            self._text(
                surface,
                str(algorithm).upper().replace("_", " "),
                (self.PADDING, 38),
                self.theme.purple,
                10,
                bold=True,
            )

        tab_width = 104 if compact else 122
        gap = 7
        total_width = len(ObservatoryTab) * tab_width + (len(ObservatoryTab) - 1) * gap
        start_x = max(190 if compact else 280, (rect.width - total_width) // 2)
        if start_x + total_width > rect.right - self.PADDING:
            start_x = rect.right - self.PADDING - total_width
        tab_rects: dict[ObservatoryTab, pygame.Rect] = {}
        for index, tab in enumerate(ObservatoryTab):
            tab_rect = pygame.Rect(start_x + index * (tab_width + gap), 17, tab_width, 40)
            selected = tab is self.active_tab
            pygame.draw.rect(
                surface,
                self.theme.panel_alt if selected else self.theme.header,
                tab_rect,
                border_radius=9,
            )
            pygame.draw.rect(
                surface,
                self.theme.blue if selected else self.theme.grid,
                tab_rect,
                1 if not selected else 2,
                border_radius=9,
            )
            label = self._font(11, bold=selected).render(tab.value, True, self.theme.text if selected else self.theme.muted)
            surface.blit(label, label.get_rect(center=tab_rect.center))
            tab_rects[tab] = tab_rect

        status_rect = pygame.Rect(
            start_x + total_width + 12,
            0,
            max(0, rect.right - self.PADDING - start_x - total_width - 12),
            rect.height,
        )
        speed_rects: tuple[pygame.Rect, ...] = ()
        if status_rect.width >= 105:
            speed_rects = self._draw_speed_status(surface, status_rect, telemetry)
        return tab_rects, speed_rects

    def _draw_speed_status(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
    ) -> tuple[pygame.Rect, ...]:
        hint = "[ ]  1-7  CLICK TO SET SPEED" if rect.width >= 205 else "[ ]  1-7 SPEED"
        hint_image = self._font(8).render(hint, True, self.theme.muted)
        surface.blit(hint_image, (rect.right - hint_image.get_width(), 8))

        speed = _number(
            _first(
                telemetry,
                "simulation_fps_target",
                "speed_target_fps",
                "decisions_per_second",
                "speed",
            )
        )
        if speed is None:
            return ()
        raw_label = _first(telemetry, "speed_label", "speed_mode")
        label = "CUSTOM" if raw_label in (_MISSING, None) else str(raw_label).upper()
        measured = _number(_first(telemetry, "simulation_fps_actual"))
        if measured is not None and rect.width >= 190:
            value_text = f"{label}  ·  sim {measured:.1f} / {int(speed)} FPS"
        else:
            value_text = f"{label}  ·  target {int(speed)} sim FPS"
        value_color = self.theme.yellow if label == "MAX" else self.theme.cyan
        value_image = self._font(10, bold=True).render(value_text, True, value_color)
        surface.blit(value_image, (rect.right - value_image.get_width(), 27))

        count = _number(_first(telemetry, "speed_preset_count"))
        active = _number(_first(telemetry, "speed_preset_index"))
        if rect.width < 155 or count is None or active is None or count < 2:
            return ()
        preset_count = max(2, int(count))
        active_index = max(0, min(preset_count - 1, int(active)))
        gap = 3
        slow_image = self._font(7, bold=True).render("SLOW", True, self.theme.muted)
        max_image = self._font(7, bold=True).render("MAX", True, self.theme.muted)
        available_track = rect.width - slow_image.get_width() - max_image.get_width() - 18
        track_width = min(142, max(82, available_track))
        segment_width = max(3, (track_width - gap * (preset_count - 1)) // preset_count)
        actual_width = segment_width * preset_count + gap * (preset_count - 1)
        total_width = slow_image.get_width() + actual_width + max_image.get_width() + 12
        label_x = rect.right - total_width
        start_x = label_x + slow_image.get_width() + 6
        surface.blit(slow_image, (label_x, 50))
        surface.blit(max_image, (start_x + actual_width + 6, 50))
        hitboxes: list[pygame.Rect] = []
        for index in range(preset_count):
            segment = pygame.Rect(start_x + index * (segment_width + gap), 51, segment_width, 6)
            hitboxes.append(
                pygame.Rect(segment.x, segment.y - 5, segment.width, segment.height + 10)
            )
            color = self.theme.grid
            if index < active_index:
                color = self.theme.blue
            elif index == active_index:
                color = value_color
            pygame.draw.rect(surface, color, segment, border_radius=3)
        return tuple(hitboxes)

    def _draw_game_tab(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
        history: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]],
        game_surface: Optional[pygame.Surface],
    ) -> None:
        gap = 14
        side_width = max(330, min(430, round(rect.width * 0.38)))
        game_rect = pygame.Rect(rect.x, rect.y, max(1, rect.width - side_width - gap), rect.height)
        side_rect = pygame.Rect(game_rect.right + gap, rect.y, side_width, rect.height)
        self._panel(surface, game_rect)
        self._panel(surface, side_rect)

        viewport = game_rect.inflate(-18, -18)
        if game_surface is None:
            self._empty_state(surface, viewport, "Game view unavailable", "Pass game_surface=... to render the live board")
        else:
            self._blit_contain(surface, game_surface, viewport)

        inner = side_rect.inflate(-18, -18)
        y = inner.y
        self._section_title(surface, "LIVE DECISION", inner.x, y)
        y += 28
        y = self._draw_metric_cards(surface, pygame.Rect(inner.x, y, inner.width, 102), telemetry, columns=3)
        y += 9
        combat_rect = pygame.Rect(inner.x, y, inner.width, 74)
        self._draw_combat_telemetry(surface, combat_rect, telemetry, compact=True)
        y = combat_rect.bottom + 9
        q_rect = pygame.Rect(inner.x, y, inner.width, min(176, max(118, inner.bottom - y - 108)))
        self._draw_q_bars(surface, q_rect, telemetry, compact=True)
        y = q_rect.bottom + 10
        memory_rect = pygame.Rect(inner.x, y, inner.width, max(72, inner.bottom - y))
        self._draw_memory(surface, memory_rect, telemetry, history)

    def _draw_vision_tab(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
    ) -> None:
        """Render every value in the live 32-feature observation contract."""

        self._panel(surface, rect)
        inner = rect.inflate(-18, -18)
        observation = _observation_values(telemetry)
        if not observation:
            self._empty_state(
                surface,
                inner,
                "Vision telemetry unavailable",
                "Provide observation as a named mapping or a vector with observation_labels",
            )
            return

        received = sum(value is not None for value in observation.values())
        self._text(surface, "AGENT VISION", (inner.x, inner.y), self.theme.text, 16, bold=True)
        self._text(
            surface,
            f"{received} / 32 live inputs received  ·  normalized agent state",
            (inner.x + 132, inner.y + 4),
            self.theme.muted,
            9,
        )
        chosen = _chosen_action_index(telemetry, _action_labels(telemetry, 4))
        action_labels = _action_labels(telemetry, 4)
        if chosen is not None:
            action_text = f"NEXT  {action_labels[chosen]}"
            image = self._font(10, bold=True).render(action_text, True, self.theme.yellow)
            surface.blit(image, (inner.right - image.get_width(), inner.y + 2))

        body = pygame.Rect(inner.x, inner.y + 36, inner.width, inner.height - 36)
        gap = 12
        left_width = max(330, round(body.width * 0.66))
        left = pygame.Rect(body.x, body.y, left_width, body.height)
        right = pygame.Rect(left.right + gap, body.y, max(1, body.right - left.right - gap), body.height)

        direction_groups = (
            ("PATHS", "path", "walkable", self.theme.blue),
            ("PELLETS", "pellet", "proximity", self.theme.yellow),
            ("POWER PELLETS", "power pellet", "proximity", self.theme.purple),
            ("THREATS", "threat", "ghost + projectile proximity", self.theme.red),
            ("EDIBLE GHOSTS", "edible ghost", "target proximity", self.theme.cyan),
        )
        row_gap = 7
        row_height = max(68, (left.height - row_gap * (len(direction_groups) - 1)) // len(direction_groups))
        for index, (title, prefix, detail, color) in enumerate(direction_groups):
            group_rect = pygame.Rect(
                left.x,
                left.y + index * (row_height + row_gap),
                left.width,
                row_height,
            )
            self._draw_directional_observation_group(
                surface,
                group_rect,
                observation,
                title=title,
                prefix=prefix,
                detail=detail,
                color=color,
                chosen_action=chosen,
            )

        heading_height = min(190, max(150, round(right.height * 0.35)))
        heading = pygame.Rect(right.x, right.y, right.width, heading_height)
        context = pygame.Rect(
            right.x,
            heading.bottom + gap,
            right.width,
            max(1, right.bottom - heading.bottom - gap),
        )
        self._draw_heading_observations(surface, heading, observation)
        self._draw_context_observations(surface, context, observation)

    def _draw_directional_observation_group(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        observation: Mapping[str, Optional[float]],
        *,
        title: str,
        prefix: str,
        detail: str,
        color: Color,
        chosen_action: Optional[int],
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=9)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=9)
        pygame.draw.rect(surface, color, pygame.Rect(rect.x, rect.y, 4, rect.height), border_radius=3)

        label_width = min(124, max(82, rect.width // 5))
        self._text(surface, title, (rect.x + 13, rect.y + 11), self.theme.text, 9, bold=True)
        self._text(surface, detail, (rect.x + 13, rect.y + 31), self.theme.muted, 8)

        directions = ("ahead", "right", "left", "reverse")
        gap = 6
        cells_x = rect.x + label_width
        cell_width = max(1, (rect.right - cells_x - 9 - gap * 3) // 4)
        for index, direction in enumerate(directions):
            cell = pygame.Rect(
                cells_x + index * (cell_width + gap),
                rect.y + 7,
                cell_width,
                rect.height - 14,
            )
            selected = index == chosen_action
            pygame.draw.rect(surface, self.theme.panel, cell, border_radius=7)
            pygame.draw.rect(
                surface,
                self.theme.yellow if selected else self.theme.grid_bright,
                cell,
                2 if selected else 1,
                border_radius=7,
            )
            value = observation.get(f"{prefix} {direction}")
            label_color = self.theme.yellow if selected else self.theme.muted
            self._text(surface, direction.upper(), (cell.x + 7, cell.y + 6), label_color, 7, bold=selected)
            value_text = _format_observation(value)
            self._text(surface, value_text, (cell.x + 7, cell.y + 20), self.theme.text, 11, bold=True)
            track = pygame.Rect(cell.x + 7, cell.bottom - 11, max(1, cell.width - 14), 5)
            pygame.draw.rect(surface, self.theme.grid, track, border_radius=3)
            if value is not None:
                fill = track.copy()
                fill.width = round(track.width * max(0.0, min(1.0, value)))
                if fill.width:
                    pygame.draw.rect(surface, color, fill, border_radius=3)

    def _draw_heading_observations(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        observation: Mapping[str, Optional[float]],
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=9)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=9)
        self._section_title(surface, "HEADING · WORLD AXIS", rect.x + 11, rect.y + 9)

        center = (rect.centerx, rect.centery + 10)
        reach_x = min(70, max(42, rect.width // 5))
        reach_y = min(47, max(35, rect.height // 4))
        positions = {
            "up": (center[0], center[1] - reach_y),
            "right": (center[0] + reach_x, center[1]),
            "down": (center[0], center[1] + reach_y),
            "left": (center[0] - reach_x, center[1]),
        }
        pygame.draw.line(surface, self.theme.grid_bright, positions["up"], positions["down"], 2)
        pygame.draw.line(surface, self.theme.grid_bright, positions["left"], positions["right"], 2)
        pygame.draw.circle(surface, self.theme.panel, center, 8)
        pygame.draw.circle(surface, self.theme.grid_bright, center, 8, 1)

        for direction, position in positions.items():
            value = observation.get(f"heading {direction}")
            active = value is not None and value >= 0.5
            fill = self.theme.green if active else self.theme.panel
            border = self.theme.green if active else self.theme.grid_bright
            pygame.draw.circle(surface, fill, position, 21)
            pygame.draw.circle(surface, border, position, 21, 2 if active else 1)
            foreground = self.theme.background if active else self.theme.text
            label = self._font(8, bold=True).render(
                direction[:1].upper(),
                True,
                foreground,
            )
            surface.blit(label, label.get_rect(center=(position[0], position[1] - 4)))
            value_color = self.theme.background if active else self.theme.muted
            value_image = self._font(7, bold=True).render(
                _format_observation(value),
                True,
                value_color,
            )
            surface.blit(value_image, value_image.get_rect(center=(position[0], position[1] + 7)))

    def _draw_context_observations(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        observation: Mapping[str, Optional[float]],
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=9)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=9)
        self._section_title(surface, "CONTEXT · NORMALIZED", rect.x + 11, rect.y + 9)

        labels = (
            "position x",
            "position y",
            "frightened time",
            "pellets remaining",
            "lives",
            "level",
            "chase mode",
            "ghosts released",
        )
        gap = 6
        top = rect.y + 31
        cell_width = max(1, (rect.width - 22 - gap) // 2)
        cell_height = max(36, (rect.bottom - top - 9 - gap * 3) // 4)
        colors = (
            self.theme.blue,
            self.theme.blue,
            self.theme.purple,
            self.theme.yellow,
            self.theme.green,
            self.theme.cyan,
            self.theme.red,
            self.theme.orange,
        )
        for index, (label, color) in enumerate(zip(labels, colors)):
            column = index % 2
            row = index // 2
            cell = pygame.Rect(
                rect.x + 11 + column * (cell_width + gap),
                top + row * (cell_height + gap),
                cell_width,
                cell_height,
            )
            pygame.draw.rect(surface, self.theme.panel, cell, border_radius=6)
            value = observation.get(label)
            self._text(surface, label.upper(), (cell.x + 7, cell.y + 5), self.theme.muted, 7, bold=True)
            value_text = _format_observation(value)
            image = self._font(9, bold=True).render(value_text, True, self.theme.text)
            surface.blit(image, (cell.right - image.get_width() - 7, cell.y + 4))
            track = pygame.Rect(cell.x + 7, cell.bottom - 9, max(1, cell.width - 14), 4)
            pygame.draw.rect(surface, self.theme.grid, track, border_radius=2)
            if value is not None:
                fill = track.copy()
                fill.width = round(track.width * max(0.0, min(1.0, value)))
                if fill.width:
                    pygame.draw.rect(surface, color, fill, border_radius=2)

    def _draw_metrics_tab(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
        history: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]],
    ) -> None:
        self._panel(surface, rect)
        inner = rect.inflate(-18, -18)
        cards_height = 90
        self._draw_metric_cards(
            surface,
            pygame.Rect(inner.x, inner.y, inner.width, cards_height),
            telemetry,
            columns=6,
        )

        health_rect = pygame.Rect(inner.x, inner.y + cards_height + 10, inner.width, 66)
        self._draw_health_strip(surface, health_rect, telemetry)
        combat_rect = pygame.Rect(inner.x, health_rect.bottom + 9, inner.width, 62)
        self._draw_combat_telemetry(surface, combat_rect, telemetry, compact=False)
        chart_top = combat_rect.bottom + 12
        bottom_height = 108
        charts_rect = pygame.Rect(inner.x, chart_top, inner.width, max(100, inner.bottom - chart_top - bottom_height - 12))
        gap = 12
        chart_width = (charts_rect.width - gap) // 2
        chart_height = (charts_rect.height - gap) // 2
        chart_specs = (
            ("REWARD", ("rewards", "reward", "episode_returns", "episode_return"), self.theme.green),
            ("LOSS", ("losses", "loss"), self.theme.orange),
            ("SCORE", ("scores", "score"), self.theme.blue),
            ("EPSILON", ("epsilons", "epsilon"), self.theme.purple),
        )
        for index, (label, aliases, color) in enumerate(chart_specs):
            chart = pygame.Rect(
                charts_rect.x + (index % 2) * (chart_width + gap),
                charts_rect.y + (index // 2) * (chart_height + gap),
                chart_width,
                chart_height,
            )
            values = _history_values(history, aliases)[-self.chart_window :]
            self._draw_chart(surface, chart, label, values, color)

        bottom = pygame.Rect(inner.x, charts_rect.bottom + 12, inner.width, max(84, inner.bottom - charts_rect.bottom - 12))
        left = pygame.Rect(bottom.x, bottom.y, max(1, round(bottom.width * 0.58) - 6), bottom.height)
        right = pygame.Rect(left.right + 12, bottom.y, max(1, bottom.right - left.right - 12), bottom.height)
        self._draw_memory(surface, left, telemetry, history)
        self._draw_replay_meter(surface, right, telemetry)

    def _draw_health_strip(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
    ) -> None:
        """Render the compact, stable learner-health telemetry contract."""

        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=9)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=9)
        health = _to_plain(telemetry.get("health"))
        if not isinstance(health, Mapping):
            self._section_title(surface, "LEARNER HEALTH", rect.x + 11, rect.y + 8)
            self._text(
                surface,
                "Health telemetry unavailable",
                (rect.x + 11, rect.y + 35),
                self.theme.muted,
                9,
            )
            return

        status = str(health.get("status", "unknown")).strip().lower()
        status_colors = {
            "healthy": self.theme.green,
            "warming_up": self.theme.blue,
            "warning": self.theme.yellow,
            "critical": self.theme.red,
        }
        status_color = status_colors.get(status, self.theme.muted)
        status_label = status.upper().replace("_", " ")
        self._text(surface, "LEARNER HEALTH", (rect.x + 11, rect.y + 8), self.theme.text, 8, bold=True)
        pill = pygame.Rect(rect.x + 112, rect.y + 6, 94, 18)
        pygame.draw.rect(surface, status_color, pill, border_radius=9)
        pill_image = self._font(7, bold=True).render(status_label, True, self.theme.background)
        surface.blit(pill_image, pill_image.get_rect(center=pill.center))

        replay = _to_plain(health.get("replay"))
        optimization = _to_plain(health.get("optimization"))
        values = _to_plain(health.get("values"))
        replay = replay if isinstance(replay, Mapping) else {}
        optimization = optimization if isinstance(optimization, Mapping) else {}
        values = values if isinstance(values, Mapping) else {}

        replay_size = _number(replay.get("size"))
        warmup = _number(replay.get("warmup_threshold"))
        ready = bool(replay.get("ready", False))
        updates = _number(optimization.get("updates"))
        update_ratio = _number(optimization.get("update_to_decision_ratio"))
        clip_ratio = _number(optimization.get("clip_ratio"))
        clip_pressure = _number(optimization.get("gradient_to_clip_ratio"))
        clip_history_complete = bool(
            optimization.get("clip_history_complete", clip_ratio is not None)
        )
        clip_fraction = _number(optimization.get("recent_clip_fraction"))
        q_abs = _number(values.get("q_abs_max"))
        td_abs = _number(values.get("td_error_abs_mean"))

        replay_text = "REPLAY  —"
        if replay_size is not None and warmup is not None:
            replay_text = f"REPLAY  {int(replay_size):,}/{int(warmup):,}  {'READY' if ready else 'WARMUP'}"
        update_text = "UPDATES  —"
        if updates is not None and update_ratio is not None:
            update_text = f"UPDATES  {int(updates):,}  ·  {update_ratio:.1%}/decision"
        clip_text = "CLIP  —"
        if clip_ratio is not None and clip_pressure is not None:
            clip_text = f"CLIP  {clip_ratio:.0%}  ·  norm {clip_pressure:.2f}×"
        elif not clip_history_complete and clip_pressure is not None:
            clip_text = f"CLIP  HISTORY N/A  ·  norm {clip_pressure:.2f}×"
        elif clip_fraction is not None:
            clip_text = f"CLIP  recent {clip_fraction:.0%}"
        value_text = "VALUES  —"
        if q_abs is not None and td_abs is not None:
            value_text = f"|Q|max {q_abs:.2f}  ·  |TD|μ {td_abs:.2f}"

        cells = (
            (replay_text, self.theme.green if ready else self.theme.blue),
            (update_text, self.theme.cyan),
            (
                clip_text,
                self.theme.orange
                if (clip_ratio or 0.0) >= 0.5 or (clip_pressure or 0.0) >= 1.0
                else self.theme.muted,
            ),
            (value_text, self.theme.purple),
        )
        start_x = rect.x + 11
        available = rect.width - 22
        cell_width = max(120, available // len(cells))
        for index, (label, color) in enumerate(cells):
            x = start_x + index * cell_width
            self._text(surface, label, (x, rect.y + 31), color, 8, bold=True)

        raw_alerts = health.get("alerts", ())
        alerts = list(raw_alerts) if isinstance(raw_alerts, Sequence) and not isinstance(raw_alerts, str) else []
        if alerts:
            alert_text = "ALERT  " + " · ".join(str(item).replace("_", " ") for item in alerts[:2])
            image = self._font(7, bold=True).render(alert_text.upper(), True, status_color)
            surface.blit(image, (rect.right - image.get_width() - 11, rect.y + 9))

    def _draw_network_tab(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
    ) -> None:
        self._panel(surface, rect)
        inner = rect.inflate(-18, -18)
        network = _network_payload(telemetry)
        layers = _network_layers(network)
        weights = _weight_matrices(network, layers)
        if not layers:
            self._empty_state(
                surface,
                inner,
                "Network telemetry unavailable",
                "Provide network.layers (or layer_names + activations) and network.weights",
            )
            return

        q_height = 150 if inner.height >= 510 else 125
        graph_rect = pygame.Rect(inner.x, inner.y, inner.width, max(180, inner.height - q_height - 12))
        q_rect = pygame.Rect(inner.x, graph_rect.bottom + 12, inner.width, max(90, inner.bottom - graph_rect.bottom - 12))
        self._draw_network_graph(surface, graph_rect, telemetry, network, layers, weights)
        self._draw_q_bars(surface, q_rect, telemetry, compact=False)

    def _draw_combat_telemetry(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
        *,
        compact: bool,
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=9)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=9)
        self._text(
            surface,
            "COMBAT TELEMETRY",
            (rect.x + 11, rect.y + 8),
            self.theme.text,
            8,
            bold=True,
        )
        caption = (
            "diagnostic · shares THREAT inputs"
            if compact
            else "diagnostic · projectile rays share the four THREAT inputs"
        )
        self._text(surface, caption, (rect.x + 119, rect.y + 8), self.theme.muted, 7)

        projectile_data = _to_plain(telemetry.get("projectiles"))
        if not isinstance(projectile_data, Mapping):
            self._text(
                surface,
                "No projectile telemetry received",
                (rect.x + 11, rect.y + 32),
                self.theme.muted,
                9,
            )
            return

        weapons = _to_plain(projectile_data.get("weapons"))
        weapons = weapons if isinstance(weapons, Mapping) else {}

        def weapon_text(owner: str, label: str) -> str:
            weapon = _to_plain(weapons.get(owner))
            if not isinstance(weapon, Mapping):
                return f"{label}  —"
            unlocked = bool(weapon.get("unlocked", False))
            early = bool(weapon.get("unlocked_early", False))
            cooldown = _number(weapon.get("cooldown_seconds")) or 0.0
            range_tiles = int(_number(weapon.get("range_tiles")) or 0)
            if not unlocked:
                status = "LOCKED"
            elif cooldown > 0:
                status = f"{'EARLY ' if early else ''}{cooldown:.1f}s"
            elif early:
                status = "EARLY READY"
            else:
                status = "READY"
            return f"{label}  {status} · {range_tiles}t"

        fire = weapon_text("BLINKY", "FIRE")
        freeze = weapon_text("INKY", "FREEZE")
        active = int(_number(projectile_data.get("active_count")) or 0)
        shots = int(_number(projectile_data.get("shots_fired")) or 0)
        fire_hits = int(_number(projectile_data.get("fireball_hits")) or 0)
        freeze_hits = int(_number(projectile_data.get("freeze_ball_hits")) or 0)
        slowed = bool(projectile_data.get("player_slowed", False))
        slow_fraction = (_number(projectile_data.get("slow_fraction")) or 0.0) * 100
        slow_timer = _number(projectile_data.get("slow_timer")) or 0.0

        midpoint = rect.x + rect.width // 2
        self._text(surface, fire, (rect.x + 11, rect.y + 29), self.theme.orange, 8, bold=True)
        self._text(surface, freeze, (midpoint, rect.y + 29), self.theme.cyan, 8, bold=True)
        if compact:
            diagnostic = f"ACTIVE {active} · SHOTS {shots} · HITS {fire_hits}/{freeze_hits}"
            player = f"PACMAN  -{slow_fraction:.0f}% · {slow_timer:.1f}s" if slowed else "PACMAN  NORMAL"
            self._text(surface, diagnostic, (rect.x + 11, rect.y + 51), self.theme.muted, 7)
            player_image = self._font(7, bold=True).render(
                player,
                True,
                self.theme.cyan if slowed else self.theme.green,
            )
            surface.blit(player_image, (rect.right - player_image.get_width() - 11, rect.y + 51))
        else:
            diagnostic = f"ACTIVE {active}    TOTAL SHOTS {shots}    FIRE HITS {fire_hits}    FREEZE HITS {freeze_hits}"
            player = f"PACMAN SLOWED {slow_fraction:.0f}% · {slow_timer:.1f}s" if slowed else "PACMAN NORMAL SPEED"
            self._text(surface, diagnostic, (rect.x + 11, rect.y + 47), self.theme.muted, 8)
            player_image = self._font(8, bold=True).render(
                player,
                True,
                self.theme.cyan if slowed else self.theme.green,
            )
            surface.blit(player_image, (rect.right - player_image.get_width() - 11, rect.y + 47))

    def _draw_metric_cards(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
        *,
        columns: int,
    ) -> int:
        fields = (
            ("SCORE", ("score",), "integer"),
            ("EPISODE", ("episode", "episodes", "games"), "integer"),
            ("REWARD", ("reward", "episode_reward", "episode_return"), "signed"),
            ("LOSS", ("loss",), "decimal"),
            ("EPSILON", ("epsilon",), "decimal"),
            ("REPLAY", ("replay_size", "memory", "replay"), "integer"),
        )
        columns = max(1, min(columns, len(fields)))
        rows = math.ceil(len(fields) / columns)
        gap = 7
        cell_width = max(1, (rect.width - gap * (columns - 1)) // columns)
        cell_height = max(42, (rect.height - gap * (rows - 1)) // rows)
        for index, (label, aliases, format_kind) in enumerate(fields):
            column = index % columns
            row = index // columns
            card = pygame.Rect(
                rect.x + column * (cell_width + gap),
                rect.y + row * (cell_height + gap),
                cell_width,
                cell_height,
            )
            pygame.draw.rect(surface, self.theme.panel_alt, card, border_radius=8)
            pygame.draw.rect(surface, self.theme.grid, card, 1, border_radius=8)
            raw = _first(telemetry, *aliases)
            value = _format_metric(raw, format_kind)
            self._text(surface, label, (card.x + 9, card.y + 7), self.theme.muted, 9, bold=True)
            value_color = self.theme.text
            number = _number(raw)
            if label == "REWARD" and number is not None:
                value_color = self.theme.green if number > 0 else self.theme.red if number < 0 else self.theme.text
            self._text(surface, value, (card.x + 9, card.y + 25), value_color, 14, bold=True)
        return rect.bottom

    def _draw_q_bars(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
        *,
        compact: bool,
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=9)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=9)
        self._section_title(surface, "ACTION VALUES", rect.x + 11, rect.y + 9)
        online = _numeric_vector(_first(telemetry, "online_q_values", "q_values", "online_q"))
        target = _numeric_vector(_first(telemetry, "target_q_values", "target_q"))
        count = max(len(online), len(target))
        if count == 0:
            self._empty_state(surface, rect.inflate(-16, -28), "No Q-values received", "online_q_values / target_q_values")
            return
        labels = _action_labels(telemetry, count)
        chosen = _chosen_action_index(telemetry, labels)
        finite = [abs(value) for value in (*online, *target) if value is not None and math.isfinite(value)]
        scale = max(finite, default=0.0)
        top = rect.y + 34
        available = max(18, rect.bottom - top - 8)
        row_height = max(18, min(31 if not compact else 27, available // count))
        label_width = min(105, max(58, rect.width // 5))
        value_width = 92 if rect.width >= 520 else 69
        track_x = rect.x + 11 + label_width
        track_width = max(40, rect.width - label_width - value_width - 26)
        center_x = track_x + track_width // 2
        for index in range(count):
            row_y = top + index * row_height
            if row_y + row_height > rect.bottom:
                break
            selected = chosen == index
            label_color = self.theme.yellow if selected else self.theme.text
            self._text(surface, labels[index], (rect.x + 11, row_y + 4), label_color, 10, bold=selected)
            track = pygame.Rect(track_x, row_y + 3, track_width, max(12, row_height - 8))
            pygame.draw.rect(surface, self.theme.grid, track, border_radius=4)
            pygame.draw.line(surface, self.theme.muted, (center_x, track.y + 1), (center_x, track.bottom - 1), 1)
            self._signed_bar(surface, track, online[index] if index < len(online) else None, scale, self.theme.cyan, 0)
            self._signed_bar(surface, track, target[index] if index < len(target) else None, scale, self.theme.magenta, 1)
            if selected:
                pygame.draw.rect(surface, self.theme.yellow, track, 1, border_radius=4)
            online_text = _format_number(online[index] if index < len(online) else None, signed=True)
            target_text = _format_number(target[index] if index < len(target) else None, signed=True)
            self._text(
                surface,
                f"{online_text} / {target_text}",
                (track.right + 7, row_y + 4),
                self.theme.muted,
                9,
            )
        legend = "online cyan  /  target magenta"
        legend_image = self._font(8).render(legend, True, self.theme.muted)
        surface.blit(legend_image, (rect.right - legend_image.get_width() - 10, rect.y + 11))

    def _signed_bar(
        self,
        surface: pygame.Surface,
        track: pygame.Rect,
        value: Optional[float],
        scale: float,
        color: Color,
        lane: int,
    ) -> None:
        if value is None or not math.isfinite(value):
            return
        half = max(1, track.width // 2 - 2)
        magnitude = 0 if scale == 0 else round(half * abs(value) / scale)
        if value != 0 and magnitude == 0:
            magnitude = 1
        lane_height = max(2, (track.height - 4) // 2)
        y = track.y + 2 + lane * lane_height
        center = track.centerx
        bar = pygame.Rect(center if value >= 0 else center - magnitude, y, magnitude, lane_height)
        if bar.width:
            pygame.draw.rect(surface, color, bar, border_radius=2)

    def _draw_chart(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        values: Sequence[Optional[float]],
        color: Color,
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=8)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=8)
        self._section_title(surface, label, rect.x + 10, rect.y + 8)
        plot = pygame.Rect(rect.x + 42, rect.y + 30, max(20, rect.width - 52), max(20, rect.height - 43))
        pygame.draw.line(surface, self.theme.grid, plot.bottomleft, plot.bottomright, 1)
        pygame.draw.line(surface, self.theme.grid, plot.topleft, plot.bottomleft, 1)
        finite = [(index, value) for index, value in enumerate(values) if value is not None and math.isfinite(value)]
        if not finite:
            self._text(surface, "No samples received", (plot.x + 8, plot.centery - 5), self.theme.muted, 9)
            return
        minimum = min(value for _, value in finite)
        maximum = max(value for _, value in finite)
        span = maximum - minimum
        if span == 0:
            padding = max(0.5, abs(maximum) * 0.08)
            minimum -= padding
            maximum += padding
            span = maximum - minimum
        if minimum < 0 < maximum:
            zero_y = plot.bottom - round((0 - minimum) / span * plot.height)
            pygame.draw.line(surface, self.theme.grid_bright, (plot.x, zero_y), (plot.right, zero_y), 1)
        denominator = max(1, len(values) - 1)
        segments: list[list[tuple[int, int]]] = []
        segment: list[tuple[int, int]] = []
        for index, value in enumerate(values):
            if value is None or not math.isfinite(value):
                if segment:
                    segments.append(segment)
                    segment = []
                continue
            x = plot.x + round(index / denominator * plot.width)
            y = plot.bottom - round((value - minimum) / span * plot.height)
            segment.append((x, y))
        if segment:
            segments.append(segment)
        for points in segments:
            if len(points) > 1:
                pygame.draw.lines(surface, color, False, points, 2)
            else:
                pygame.draw.circle(surface, color, points[0], 2)
        self._text(surface, _format_number(maximum), (rect.x + 7, plot.y - 2), self.theme.muted, 8)
        self._text(surface, _format_number(minimum), (rect.x + 7, plot.bottom - 8), self.theme.muted, 8)
        count_text = f"{len(finite)} sample{'s' if len(finite) != 1 else ''}"
        image = self._font(8).render(count_text, True, self.theme.muted)
        surface.blit(image, (plot.right - image.get_width(), rect.y + 9))

    def _draw_memory(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
        history: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]],
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=8)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=8)
        self._section_title(surface, "RECENT REPLAY MEMORY", rect.x + 10, rect.y + 8)
        rewards = _memory_values(telemetry, history, ("recent_rewards", "rewards"), ("reward",))
        actions = _memory_raw_values(telemetry, history, ("recent_actions", "actions"), ("action", "action_index"))
        dones = _memory_raw_values(telemetry, history, ("recent_dones", "dones"), ("done", "terminal"))
        count = max(len(rewards), len(actions), len(dones))
        if count == 0:
            self._text(surface, "No transitions received", (rect.x + 11, rect.y + 35), self.theme.muted, 9)
            return
        max_slots = max(1, min(36, (rect.width - 20) // 12))
        start = max(0, count - max_slots)
        visible_count = count - start
        strip = pygame.Rect(rect.x + 10, rect.y + 33, rect.width - 20, min(32, max(19, rect.height - 48)))
        gap = 3
        slot_width = max(5, (strip.width - gap * (visible_count - 1)) // visible_count)
        for visible_index, source_index in enumerate(range(start, count)):
            reward = rewards[source_index] if source_index < len(rewards) else None
            action = actions[source_index] if source_index < len(actions) else None
            terminal = bool(dones[source_index]) if source_index < len(dones) and dones[source_index] is not None else False
            color = self.theme.grid_bright
            if reward is not None:
                color = self.theme.green if reward > 0 else self.theme.red if reward < 0 else self.theme.blue
            slot = pygame.Rect(strip.x + visible_index * (slot_width + gap), strip.y, slot_width, strip.height)
            pygame.draw.rect(surface, color, slot, border_radius=3)
            if terminal:
                pygame.draw.rect(surface, self.theme.yellow, slot, 2, border_radius=3)
            if action is not None and slot_width >= 12:
                action_text = str(action)
                if len(action_text) > 2:
                    action_text = action_text[:2]
                image = self._font(8, bold=True).render(action_text, True, self.theme.background)
                surface.blit(image, image.get_rect(center=slot.center))
        if rect.height >= 78:
            self._text(
                surface,
                "green +reward   red −reward   yellow terminal border",
                (rect.x + 10, strip.bottom + 7),
                self.theme.muted,
                8,
            )

    def _draw_replay_meter(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=8)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=8)
        self._section_title(surface, "REPLAY CAPACITY", rect.x + 10, rect.y + 8)
        size_raw = _first(telemetry, "replay_size", "memory", "replay")
        capacity_raw = _first(telemetry, "replay_capacity", "memory_capacity", "capacity")
        size = _number(size_raw)
        capacity = _number(capacity_raw)
        if size is None or capacity is None or capacity <= 0:
            self._text(surface, "Size / capacity unavailable", (rect.x + 10, rect.y + 35), self.theme.muted, 9)
            return
        fraction = max(0.0, min(1.0, size / capacity))
        track = pygame.Rect(rect.x + 10, rect.y + 38, rect.width - 20, 12)
        pygame.draw.rect(surface, self.theme.grid, track, border_radius=6)
        fill = track.copy()
        fill.width = round(track.width * fraction)
        if fill.width:
            pygame.draw.rect(surface, self.theme.blue, fill, border_radius=6)
        self._text(
            surface,
            f"{int(size):,} / {int(capacity):,}   {fraction:.1%}",
            (rect.x + 10, track.bottom + 8),
            self.theme.text,
            10,
            bold=True,
        )

    def _draw_network_graph(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        telemetry: Mapping[str, Any],
        network: Mapping[str, Any],
        layers: Sequence[_Layer],
        weights: Sequence[Any],
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_alt, rect, border_radius=9)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=9)
        self._section_title(surface, "REAL NETWORK STATE", rect.x + 11, rect.y + 9)
        architecture = _numeric_vector(network.get("architecture"))
        architecture_text = " → ".join(str(int(value)) for value in architecture if value is not None)
        parameter_count = _number(network.get("parameter_count"))
        detail_parts = ["actual forward pass"]
        if architecture_text:
            detail_parts.append(architecture_text)
        if parameter_count is not None:
            detail_parts.append(f"{int(parameter_count):,} parameters")
        self._text(
            surface,
            "  ·  ".join(detail_parts),
            (rect.x + 145, rect.y + 10),
            self.theme.muted,
            9,
        )
        graph = pygame.Rect(rect.x + 28, rect.y + 48, rect.width - 56, max(90, rect.height - 86))
        layer_count = len(layers)
        x_positions = [
            graph.x + round(index * graph.width / max(1, layer_count - 1))
            for index in range(layer_count)
        ]
        sampled: list[tuple[list[int], list[Optional[float]], list[tuple[int, int]]]] = []
        node_radius = max(5, min(10, graph.height // (self.max_visible_neurons * 3)))
        for layer, x in zip(layers, x_positions):
            full_count = layer.size if layer.size is not None else len(layer.activations)
            full_count = max(0, full_count)
            indices = _sample_indices(full_count, self.max_visible_neurons)
            values = [layer.activations[index] if index < len(layer.activations) else None for index in indices]
            if indices:
                step = graph.height / max(1, len(indices) - 1)
                positions = [(x, graph.y + round(i * step)) for i in range(len(indices))]
            else:
                positions = []
            sampled.append((indices, values, positions))

        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        layout = str(network.get("weight_layout", "out_in")).lower()
        for connection_index in range(layer_count - 1):
            matrix = weights[connection_index] if connection_index < len(weights) else None
            src_indices, _, src_positions = sampled[connection_index]
            dst_indices, _, dst_positions = sampled[connection_index + 1]
            connection_values = _sampled_weights(
                matrix,
                src_indices,
                dst_indices,
                layers[connection_index].size,
                layers[connection_index + 1].size,
                layout,
            )
            finite_weights = [abs(value) for _, _, value in connection_values if math.isfinite(value)]
            weight_scale = max(finite_weights, default=0.0)
            src_position_by_index = dict(zip(src_indices, src_positions))
            dst_position_by_index = dict(zip(dst_indices, dst_positions))
            for src_index, dst_index, value in connection_values:
                start = src_position_by_index.get(src_index)
                end = dst_position_by_index.get(dst_index)
                if start is None or end is None:
                    continue
                ratio = 0.0 if weight_scale == 0 else abs(value) / weight_scale
                alpha = 24 + round(154 * ratio)
                width = 1 + round(3 * ratio)
                base = self.theme.cyan if value > 0 else self.theme.magenta if value < 0 else self.theme.muted
                pygame.draw.line(overlay, (*base, alpha), start, end, width)
        surface.blit(overlay, (0, 0))

        labels = _action_labels(telemetry, max((layer.size or len(layer.activations) for layer in layers[-1:]), default=0))
        chosen = _chosen_action_index(telemetry, labels)
        for layer_index, (layer, x, sample_data) in enumerate(zip(layers, x_positions, sampled)):
            indices, values, positions = sample_data
            finite_activations = [abs(value) for value in values if value is not None and math.isfinite(value)]
            activation_scale = max(finite_activations, default=0.0)
            for node_index, activation, position in zip(indices, values, positions):
                if activation is None or not math.isfinite(activation):
                    pygame.draw.circle(surface, self.theme.panel, position, node_radius)
                    pygame.draw.circle(surface, self.theme.muted, position, node_radius, 1)
                else:
                    ratio = 0.0 if activation_scale == 0 else min(1.0, abs(activation) / activation_scale)
                    base = self.theme.green if activation >= 0 else self.theme.orange
                    fill = _blend(self.theme.panel, base, 0.22 + 0.68 * ratio)
                    pygame.draw.circle(surface, fill, position, node_radius)
                    pygame.draw.circle(surface, base, position, node_radius, 1)
                if layer_index == layer_count - 1 and chosen == node_index:
                    pygame.draw.circle(surface, self.theme.yellow, position, node_radius + 3, 2)

            title = layer.name
            title_image = self._font(10, bold=True).render(title, True, self.theme.text)
            surface.blit(title_image, (x - title_image.get_width() // 2, graph.bottom + 12))
            display_size = layer.full_size if layer.full_size is not None else layer.size
            size_text = "size unavailable" if display_size is None else str(display_size)
            if display_size is not None and len(indices) < display_size:
                size_text = f"{len(indices)} / {display_size} shown"
            elif values and not any(value is not None for value in values):
                size_text = f"{size_text} · no activations"
            size_image = self._font(8).render(size_text, True, self.theme.muted)
            surface.blit(size_image, (x - size_image.get_width() // 2, graph.bottom + 27))

        missing_weight_count = max(0, len(layers) - 1 - len(weights))
        if missing_weight_count:
            note = f"{missing_weight_count} weight matrix{'es' if missing_weight_count != 1 else ''} unavailable"
            image = self._font(8).render(note, True, self.theme.muted)
            surface.blit(image, (rect.right - image.get_width() - 11, rect.y + 11))

    def _panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, self.theme.panel, rect, border_radius=11)
        pygame.draw.rect(surface, self.theme.grid, rect, 1, border_radius=11)

    def _section_title(self, surface: pygame.Surface, value: str, x: int, y: int) -> None:
        self._text(surface, value, (x, y), self.theme.muted, 9, bold=True)

    def _text(
        self,
        surface: pygame.Surface,
        value: str,
        position: tuple[int, int],
        color: Color,
        size: int,
        *,
        bold: bool = False,
    ) -> None:
        surface.blit(self._font(size, bold=bold).render(str(value), True, color), position)

    def _empty_state(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        title: str,
        detail: str,
    ) -> None:
        title_image = self._font(14, bold=True).render(title, True, self.theme.text)
        detail_image = self._font(9).render(detail, True, self.theme.muted)
        title_y = rect.centery - 15
        surface.blit(title_image, (rect.centerx - title_image.get_width() // 2, title_y))
        surface.blit(detail_image, (rect.centerx - detail_image.get_width() // 2, title_y + 25))

    def _blit_contain(self, destination: pygame.Surface, source: pygame.Surface, rect: pygame.Rect) -> None:
        if source.get_width() <= 0 or source.get_height() <= 0:
            self._empty_state(destination, rect, "Game view unavailable", "The supplied surface has no drawable area")
            return
        scale = min(rect.width / source.get_width(), rect.height / source.get_height())
        size = (max(1, round(source.get_width() * scale)), max(1, round(source.get_height() * scale)))
        image = pygame.transform.smoothscale(source, size)
        image_rect = image.get_rect(center=rect.center)
        destination.blit(image, image_rect)
        pygame.draw.rect(destination, self.theme.grid_bright, image_rect, 1, border_radius=4)


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return _MISSING


def _number(value: Any) -> Optional[float]:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _to_plain(value: Any) -> Any:
    if value is _MISSING:
        return value
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
    except (RuntimeError, TypeError, ValueError):
        return value
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_record_sequence(value: Any) -> bool:
    return _is_sequence(value) and (not value or all(isinstance(item, Mapping) for item in value))


def _numeric_vector(value: Any) -> list[Optional[float]]:
    value = _to_plain(value)
    if not _is_sequence(value):
        return []
    return [_number(item) for item in value]


def _observation_values(telemetry: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """Return only caller-supplied, named observation values.

    The live session supplies a mapping.  Recorded integrations may instead
    provide a vector and the matching ``observation_labels``; both forms keep
    every displayed value traceable to the telemetry payload.
    """

    raw = _to_plain(_first(telemetry, "observation", "observations", "state"))
    if isinstance(raw, Mapping):
        return {str(label): _number(value) for label, value in raw.items()}
    values = _numeric_vector(raw)
    labels = _to_plain(_first(telemetry, "observation_labels", "state_labels"))
    if not values or not _is_sequence(labels):
        return {}
    return {
        str(label): values[index]
        for index, label in enumerate(labels)
        if index < len(values)
    }


def _format_observation(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _format_number(value: Optional[float], *, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    absolute = abs(value)
    precision = 3 if absolute < 10 else 2 if absolute < 100 else 1
    return f"{value:+.{precision}f}" if signed else f"{value:.{precision}f}"


def _format_metric(value: Any, kind: str) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if kind == "integer":
        return f"{int(number):,}" if number.is_integer() else f"{number:,.1f}"
    if kind == "signed":
        return _format_number(number, signed=True)
    return _format_number(number)


def _history_values(
    history: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]],
    aliases: Sequence[str],
) -> list[Optional[float]]:
    if isinstance(history, Mapping):
        raw = _first(history, *aliases)
        return _numeric_vector(raw)
    if _is_record_sequence(history):
        values: list[Optional[float]] = []
        for record in history:
            raw = _first(record, *aliases)
            values.append(_number(raw))
        return values
    return []


def _memory_values(
    telemetry: Mapping[str, Any],
    history: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]],
    telemetry_aliases: Sequence[str],
    record_aliases: Sequence[str],
) -> list[Optional[float]]:
    raw = _first(telemetry, *telemetry_aliases)
    values = _numeric_vector(raw)
    if values:
        return values
    return _history_values(history, record_aliases)


def _memory_raw_values(
    telemetry: Mapping[str, Any],
    history: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]],
    telemetry_aliases: Sequence[str],
    record_aliases: Sequence[str],
) -> list[Any]:
    raw = _to_plain(_first(telemetry, *telemetry_aliases))
    if _is_sequence(raw):
        return list(raw)
    if _is_record_sequence(history):
        return [None if (value := _first(record, *record_aliases)) is _MISSING else _to_plain(value) for record in history]
    if isinstance(history, Mapping):
        raw = _to_plain(_first(history, *telemetry_aliases, *record_aliases))
        if _is_sequence(raw):
            return list(raw)
    return []


def _action_labels(telemetry: Mapping[str, Any], count: int) -> list[str]:
    raw = _to_plain(_first(telemetry, "action_labels", "actions"))
    supplied = [str(item) for item in raw] if _is_sequence(raw) else []
    return [supplied[index] if index < len(supplied) else f"A{index}" for index in range(count)]


def _chosen_action_index(telemetry: Mapping[str, Any], labels: Sequence[str]) -> Optional[int]:
    raw = _first(telemetry, "chosen_action", "action_index", "selected_action")
    if raw is _MISSING or raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        lowered = raw.casefold()
        for index, label in enumerate(labels):
            if label.casefold() == lowered:
                return index
        try:
            raw = int(raw)
        except ValueError:
            return None
    try:
        index = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return index if 0 <= index < len(labels) else None


def _network_payload(telemetry: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = telemetry.get("network")
    if isinstance(nested, Mapping):
        return nested
    # Supporting top-level fields keeps integration lightweight while still
    # preserving the rule that only caller-supplied values are displayed.
    keys = (
        "layers", "network_layers", "layer_names", "layer_sizes",
        "activations", "layer_activations", "weights", "network_weights",
        "weight_layout",
    )
    return {key: telemetry[key] for key in keys if key in telemetry}


def _network_layers(network: Mapping[str, Any]) -> list[_Layer]:
    raw_layers = _to_plain(_first(network, "layers", "network_layers"))
    raw_activations = _to_plain(_first(network, "activations", "layer_activations"))
    names = _to_plain(network.get("layer_names"))
    sizes = _to_plain(network.get("layer_sizes"))
    layers: list[_Layer] = []

    if _is_sequence(raw_layers):
        for index, raw_layer in enumerate(raw_layers):
            if isinstance(raw_layer, Mapping):
                name = str(raw_layer.get("name", raw_layer.get("label", f"L{index}")))
                activation_raw = _to_plain(raw_layer.get("activations", raw_layer.get("activation")))
                activations = tuple(_numeric_vector(activation_raw))
                size = _positive_int(raw_layer.get("size", raw_layer.get("width")))
                if size is None and activations:
                    size = len(activations)
                full_size = _positive_int(raw_layer.get("full_size", raw_layer.get("original_size")))
                layers.append(_Layer(name, size, activations, full_size or size))
            elif isinstance(raw_layer, str):
                activation_raw = _sequence_item(raw_activations, index)
                activations = tuple(_numeric_vector(activation_raw))
                size = _positive_int(_sequence_item(sizes, index))
                if size is None and activations:
                    size = len(activations)
                layers.append(_Layer(raw_layer, size, activations))
            else:
                size = _positive_int(raw_layer)
                if size is not None:
                    activation_raw = _sequence_item(raw_activations, index)
                    layers.append(_Layer(f"L{index}", size, tuple(_numeric_vector(activation_raw))))
        if layers:
            return layers

    activation_groups: list[Any] = []
    if isinstance(raw_activations, Mapping):
        for index, (name, activation_raw) in enumerate(raw_activations.items()):
            activations = tuple(_numeric_vector(activation_raw))
            size = _positive_int(_mapping_or_sequence_item(sizes, name, index))
            if size is None and activations:
                size = len(activations)
            layers.append(_Layer(str(name), size, activations))
        return layers
    if _is_sequence(raw_activations):
        activation_groups = list(raw_activations)

    name_list = [str(item) for item in names] if _is_sequence(names) else []
    size_list = list(sizes) if _is_sequence(sizes) else []
    count = max(len(name_list), len(size_list), len(activation_groups))
    for index in range(count):
        activations = tuple(_numeric_vector(_sequence_item(activation_groups, index)))
        size = _positive_int(_sequence_item(size_list, index))
        if size is None and activations:
            size = len(activations)
        layers.append(_Layer(name_list[index] if index < len(name_list) else f"L{index}", size, activations))
    return layers


def _weight_matrices(network: Mapping[str, Any], layers: Sequence[_Layer]) -> list[Any]:
    raw = _to_plain(_first(network, "weights", "network_weights"))
    if _is_sequence(raw):
        return list(raw)
    matrices: list[Any] = []
    raw_layers = _to_plain(_first(network, "layers", "network_layers"))
    if _is_sequence(raw_layers):
        # A layer's weights are interpreted as the incoming matrix for that layer.
        for layer in list(raw_layers)[1:]:
            if isinstance(layer, Mapping) and "weights" in layer:
                matrices.append(_to_plain(layer["weights"]))
    return matrices[: max(0, len(layers) - 1)]


def _sample_indices(count: int, limit: int) -> list[int]:
    if count <= 0:
        return []
    if count <= limit:
        return list(range(count))
    sampled = [round(index * (count - 1) / (limit - 1)) for index in range(limit)]
    return list(dict.fromkeys(sampled))


def _sampled_weights(
    matrix: Any,
    source_indices: Sequence[int],
    destination_indices: Sequence[int],
    source_size: Optional[int],
    destination_size: Optional[int],
    layout: str,
) -> list[tuple[int, int, float]]:
    matrix = _to_plain(matrix)
    if not _is_sequence(matrix):
        return []
    rows = list(matrix)
    if not rows or not all(_is_sequence(row) for row in rows):
        return []
    row_count = len(rows)
    column_count = min((len(row) for row in rows), default=0)
    if layout not in ("out_in", "in_out"):
        layout = "out_in"
    if layout == "out_in" and source_size is not None and destination_size is not None:
        if row_count < destination_size or column_count < source_size:
            if row_count >= source_size and column_count >= destination_size:
                layout = "in_out"
    values: list[tuple[int, int, float]] = []
    for source in source_indices:
        for destination in destination_indices:
            row, column = (destination, source) if layout == "out_in" else (source, destination)
            if row >= row_count or column >= len(rows[row]):
                continue
            value = _number(rows[row][column])
            if value is not None:
                values.append((source, destination, value))
    return values


def _positive_int(value: Any) -> Optional[int]:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _sequence_item(sequence: Any, index: int) -> Any:
    if _is_sequence(sequence) and index < len(sequence):
        return sequence[index]
    return None


def _mapping_or_sequence_item(container: Any, key: Any, index: int) -> Any:
    if isinstance(container, Mapping):
        return container.get(key)
    return _sequence_item(container, index)


def _blend(first: Color, second: Color, amount: float) -> Color:
    amount = max(0.0, min(1.0, amount))
    return tuple(round(a + (b - a) * amount) for a, b in zip(first, second))  # type: ignore[return-value]
