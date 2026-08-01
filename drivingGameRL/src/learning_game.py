"""Interactive Pygame controller for Driving Lab learning experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pygame

from .learning_runtime import ChampionRace, DrivingLearningSession
from .learning_visualization import (
    DrivingLearningVisualization,
    LEARNING_WINDOW_SIZE,
)
from .vehicle import DriverControls


class DrivingLearningGame:
    """Run accelerated training and fair 60-Hz champion races in one window."""

    SPEED_PRESETS = (1, 4, 16, 64, 256)

    def __init__(
        self,
        session: DrivingLearningSession,
        *,
        render: bool = True,
        learning_speed: int = 16,
        checkpoint_path: str | Path | None = None,
    ):
        pygame.init()
        pygame.font.init()
        self.session = session
        self.render_enabled = render
        if render:
            self.screen = pygame.display.set_mode(LEARNING_WINDOW_SIZE)
            pygame.display.set_caption("Late Night Driving · Learning Observatory")
        else:
            self.screen = pygame.Surface(LEARNING_WINDOW_SIZE)
        self.dashboard = DrivingLearningVisualization(session.env)
        self.speed_index = min(
            range(len(self.SPEED_PRESETS)),
            key=lambda index: abs(self.SPEED_PRESETS[index] - learning_speed),
        )
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser()
            if checkpoint_path is not None
            else Path("drivingGameRL/models/checkpoints")
            / f"{session.config.algorithm}.pth"
        )
        self.paused = False
        self.running = True
        self.race: ChampionRace | None = None
        self.training_steps = 0
        self._status = "training"
        self._last_telemetry: dict[str, Any] = {}

    @property
    def steps_per_frame(self) -> int:
        return self.SPEED_PRESETS[self.speed_index]

    @property
    def speed_label(self) -> str:
        if self.speed_index == len(self.SPEED_PRESETS) - 1:
            return "MAX"
        return f"{self.steps_per_frame}x"

    def change_speed(self, direction: int) -> int:
        self.speed_index = max(
            0, min(len(self.SPEED_PRESETS) - 1, self.speed_index + direction)
        )
        self._status = f"training speed {self.speed_label}"
        return self.steps_per_frame

    def start_race(self) -> ChampionRace:
        self.race = ChampionRace(self.session)
        self._status = "race"
        return self.race

    def leave_race(self) -> None:
        self.race = None
        self.dashboard.return_to_training_requested = False
        self._status = "paused" if self.paused else "training"

    def toggle_race(self) -> None:
        if self.race is None:
            self.start_race()
        else:
            self.leave_race()

    @staticmethod
    def controls_from_keyboard() -> DriverControls:
        keys = pygame.key.get_pressed()
        throttle = float(keys[pygame.K_w] or keys[pygame.K_UP])
        throttle -= float(keys[pygame.K_s] or keys[pygame.K_DOWN]) * 0.65
        steering = float(keys[pygame.K_d] or keys[pygame.K_RIGHT])
        steering -= float(keys[pygame.K_a] or keys[pygame.K_LEFT])
        brake = float(keys[pygame.K_SPACE])
        return DriverControls(throttle=throttle, steering=steering, brake=brake)

    def _save_checkpoint(self) -> Path:
        output = self.session.save(self.checkpoint_path)
        self._status = f"saved {output.name}"
        return output

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    continue
                if event.key == pygame.K_p:
                    self.toggle_race()
                    continue
                if self.race is not None:
                    if event.key == pygame.K_r:
                        self.start_race()
                    continue
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self._status = "paused" if self.paused else "training"
                    continue
                if event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA, pygame.K_MINUS):
                    self.change_speed(-1)
                    continue
                if event.key in (
                    pygame.K_RIGHTBRACKET,
                    pygame.K_PERIOD,
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                ):
                    self.change_speed(1)
                    continue
                if event.key == pygame.K_s:
                    self._save_checkpoint()
                    continue
                if event.key == pygame.K_r:
                    self.session.reset_current_evaluation()
                    self._status = "evaluation reset"
                    continue
                if event.key == pygame.K_n and self.paused:
                    self.session.step()
                    self.training_steps += 1
                    self._status = "single step"
                    continue
                if event.key == pygame.K_F12:
                    self.save_screenshot("driving-learning-screenshot.png")
                    continue
            self.dashboard.handle_event(event)

    def _training_telemetry(self) -> dict[str, Any]:
        data = self.session.telemetry()
        data.update(
            {
                "phase": self._status,
                "paused": self.paused,
                "training_speed": self.steps_per_frame,
                "training_speed_label": self.speed_label,
                "training_steps": self.training_steps,
            }
        )
        return data

    def _race_telemetry(self) -> dict[str, Any]:
        assert self.race is not None
        race = self.race.telemetry()
        training = self._last_telemetry or self.session.telemetry()
        race.update(
            {
                "generation": training.get("generation", 0),
                "best_fitness": training.get("best_fitness", 0.0),
                "champion_member": training.get("champion_member", 0),
                "phase": "race finished" if self.race.finished else "live race",
            }
        )
        return race

    def draw(self) -> pygame.Surface:
        if self.race is None:
            self._last_telemetry = self._training_telemetry()
            frame = self.dashboard.draw(self.session.env, self._last_telemetry)
        else:
            frame = self.dashboard.draw_race(
                self.race.human_env,
                self.race.champion_env,
                self._race_telemetry(),
            )
        self.screen.blit(frame, (0, 0))
        return self.screen

    def save_screenshot(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.draw()
        pygame.image.save(self.screen, str(output))
        return output

    def run(
        self,
        *,
        fps: int = 60,
        max_training_steps: int | None = None,
        max_generations: int | None = None,
    ) -> None:
        """Run until closed or one of the optional training limits is reached."""

        clock = pygame.time.Clock()
        starting_generations = self.session.completed_generations
        self.running = True
        while self.running:
            self.handle_events()
            if self.race is not None:
                # A race is intentionally real-time and never inherits the
                # accelerated training multiplier.
                if not self.race.finished:
                    self.race.step(self.controls_from_keyboard())
            elif not self.paused:
                for _ in range(self.steps_per_frame):
                    if (
                        max_training_steps is not None
                        and self.training_steps >= max_training_steps
                    ):
                        self.running = False
                        break
                    if (
                        max_generations is not None
                        and self.session.completed_generations - starting_generations
                        >= max_generations
                    ):
                        self.running = False
                        break
                    self.session.step()
                    self.training_steps += 1

            if self.render_enabled:
                self.draw()
                pygame.display.flip()
                clock.tick(fps)
            if (
                self.race is None
                and max_training_steps is not None
                and self.training_steps >= max_training_steps
            ):
                self.running = False
            if (
                self.race is None
                and max_generations is not None
                and self.session.completed_generations - starting_generations
                >= max_generations
            ):
                self.running = False

    def close(self) -> None:
        if self.render_enabled:
            pygame.display.quit()


__all__ = ("DrivingLearningGame",)
