"""Interactive Pygame controller for Driving Lab learning experiments."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from time import perf_counter
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
    POPULATION_CAR_PRESETS = (2, 4, 8, 12)
    DEFAULT_TRAINING_FRAME_BUDGET_MS = 12.0
    MAX_INTERACTIVE_BATCH_TICKS = 8
    SPEED_BUDGET_MULTIPLIERS = (1.0, 1.0, 1.0, 1.5, 3.0)

    def __init__(
        self,
        session: DrivingLearningSession,
        *,
        render: bool = True,
        learning_speed: int = 16,
        checkpoint_path: str | Path | None = None,
        show_sensor_rays: bool = True,
        show_population_cars: bool | None = None,
        population_car_limit: int = 8,
        training_frame_budget_ms: float = DEFAULT_TRAINING_FRAME_BUDGET_MS,
    ):
        if (
            isinstance(population_car_limit, bool)
            or not isinstance(population_car_limit, int)
            or population_car_limit not in self.POPULATION_CAR_PRESETS
        ):
            choices = ", ".join(str(value) for value in self.POPULATION_CAR_PRESETS)
            raise ValueError(f"population_car_limit must be one of: {choices}")
        if (
            isinstance(training_frame_budget_ms, bool)
            or not isinstance(training_frame_budget_ms, Real)
            or not math.isfinite(float(training_frame_budget_ms))
            or float(training_frame_budget_ms) <= 0
        ):
            raise ValueError("training_frame_budget_ms must be finite and positive")
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
        self.show_sensor_rays = bool(show_sensor_rays)
        self.show_population_cars = (
            bool(session.is_population)
            if show_population_cars is None
            else bool(show_population_cars)
        )
        self.population_car_limit = int(population_car_limit)
        self.training_frame_budget_ms = float(training_frame_budget_ms)
        self.population_rollouts: Any | None = None
        self.training_steps = 0
        self._last_training_slice_steps = 0
        self._training_slice_capped = False
        self._training_slice_ms = 0.0
        self._training_ticks_per_second = 0.0
        self._environment_decisions_per_second = 0.0
        self._estimated_training_tick_ms = self.training_frame_budget_ms
        self._frame_time_ms = 0.0
        self._render_fps = 0.0
        self._status = "training"
        self._last_telemetry: dict[str, Any] = {}
        if self.show_population_cars:
            self._ensure_population_rollouts(force=True)

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

    @property
    def effective_training_frame_budget_ms(self) -> float:
        """Responsive wall-time budget scaled for accelerated presets."""

        return (
            self.training_frame_budget_ms
            * self.SPEED_BUDGET_MULTIPLIERS[self.speed_index]
        )

    def change_population_car_limit(self, direction: int) -> int:
        """Cycle the visual rollout breadth without touching scored training."""

        index = self.POPULATION_CAR_PRESETS.index(self.population_car_limit)
        index = (index + direction) % len(self.POPULATION_CAR_PRESETS)
        self.population_car_limit = self.POPULATION_CAR_PRESETS[index]
        # A manager owns exactly the clones it displays. Recreate it only when
        # the visualization is enabled; changing C while hidden does no work.
        self.population_rollouts = None
        if self.show_population_cars:
            self._ensure_population_rollouts(force=True)
        self._status = f"generation cars limit {self.population_car_limit}"
        return self.population_car_limit

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

    def _ensure_population_rollouts(self, *, force: bool = False) -> Any:
        if self.population_rollouts is None:
            from .population_rollout import PopulationRolloutManager

            self.population_rollouts = PopulationRolloutManager(
                self.session,
                max_cars=self.population_car_limit,
            )
            force = True
        self.population_rollouts.refresh(force=force)
        return self.population_rollouts

    def toggle_population_cars(self) -> bool:
        self.show_population_cars = not self.show_population_cars
        if self.show_population_cars:
            self._ensure_population_rollouts(force=True)
        self._status = (
            "generation cars on" if self.show_population_cars else "generation cars off"
        )
        return self.show_population_cars

    def toggle_sensor_rays(self) -> bool:
        self.show_sensor_rays = not self.show_sensor_rays
        self._status = "sensor rays on" if self.show_sensor_rays else "sensor rays off"
        return self.show_sensor_rays

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
                if event.key == pygame.K_v:
                    self.toggle_sensor_rays()
                    continue
                if self.race is not None:
                    if event.key == pygame.K_r:
                        self.start_race()
                    continue
                if event.key == pygame.K_m:
                    self.toggle_population_cars()
                    continue
                if event.key == pygame.K_c:
                    self.change_population_car_limit(1)
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
                    self._last_training_slice_steps = 1
                    if self.show_population_cars:
                        self._ensure_population_rollouts().step()
                    self._status = "single step"
                    continue
                if event.key == pygame.K_F12:
                    self.save_screenshot("driving-learning-screenshot.png")
                    continue
            self.dashboard.handle_event(event)
            consume = getattr(self.dashboard, "consume_action_requests", None)
            if consume is not None:
                for action in consume():
                    self._apply_dashboard_action(str(action))

    def _apply_dashboard_action(self, action: str) -> None:
        """Apply one mouse-accessible dashboard command."""

        if action == "toggle_pause":
            if self.race is None:
                self.paused = not self.paused
                self._status = "paused" if self.paused else "training"
        elif action == "speed_down":
            if self.race is None:
                self.change_speed(-1)
        elif action == "speed_up":
            if self.race is None:
                self.change_speed(1)
        elif action == "toggle_population_cars":
            if self.race is None:
                self.toggle_population_cars()
        elif action == "toggle_sensor_rays":
            self.toggle_sensor_rays()

    def _training_telemetry(self) -> dict[str, Any]:
        data = self.session.telemetry()
        rollouts: list[dict[str, Any]] = []
        rollout_generation = self.session.current_generation
        if self.show_population_cars:
            manager = self._ensure_population_rollouts()
            rollouts = manager.telemetry(include_rays=self.show_sensor_rays)
            rollout_generation = manager.generation
        data.update(
            {
                "phase": self._status,
                "paused": self.paused,
                "training_speed": self.steps_per_frame,
                "training_speed_label": self.speed_label,
                "requested_training_steps_per_frame": self.steps_per_frame,
                "effective_training_steps_per_frame": (self._last_training_slice_steps),
                "frame_training_steps": self._last_training_slice_steps,
                "training_slice_capped": self._training_slice_capped,
                "training_frame_budget_ms": self.training_frame_budget_ms,
                "effective_training_frame_budget_ms": (
                    self.effective_training_frame_budget_ms
                ),
                "training_slice_ms": self._training_slice_ms,
                "training_ticks_per_second": self._training_ticks_per_second,
                "environment_decisions_per_second": (
                    self._environment_decisions_per_second
                ),
                "frame_time_ms": self._frame_time_ms,
                "render_fps": self._render_fps,
                "training_steps": self.training_steps,
                "show_sensor_rays": self.show_sensor_rays,
                "show_population_cars": self.show_population_cars,
                "population_rollouts": rollouts,
                "population_rollout_generation": rollout_generation,
                "population_car_limit": self.population_car_limit,
                "population_preview_limit": self.population_car_limit,
                "population_preview_count": len(rollouts),
            }
        )
        health_value = data.get("health")
        if isinstance(health_value, dict):
            health = dict(health_value)
            throughput = dict(health.get("throughput", {}))
            throughput.update(
                {
                    "decisions_per_second": max(
                        0.0, float(self._environment_decisions_per_second)
                    ),
                    "ticks_per_second": max(
                        0.0, float(self._training_ticks_per_second)
                    ),
                    "frame_time_ms": max(0.0, float(self._frame_time_ms)),
                    "render_fps": max(0.0, float(self._render_fps)),
                }
            )
            health["throughput"] = throughput
            data["health"] = health
        return data

    def _advance_training_slice(
        self,
        *,
        starting_generations: int,
        max_training_steps: int | None,
        max_generations: int | None,
    ) -> int:
        """Advance a bounded interactive slice and return its exact step count.

        The selected speed remains the requested upper bound. In a visible
        window, expensive policies yield once the small wall-clock budget is
        consumed so input and rendering stay responsive. Headless runs retain
        the original exact ``steps_per_frame`` batching and full throughput.
        This changes only where presentation frames occur; scored environment
        transitions keep the same order and termination limits.
        """

        started_at = perf_counter()
        starting_decisions = self.session.environment_decisions
        advanced = 0
        self._training_slice_capped = False
        while advanced < self.steps_per_frame:
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

            remaining = self.steps_per_frame - advanced
            if max_training_steps is not None:
                remaining = min(
                    remaining,
                    max_training_steps - self.training_steps,
                )
            if self.render_enabled:
                elapsed_ms = (perf_counter() - started_at) * 1_000.0
                budget_left = max(
                    0.0,
                    self.effective_training_frame_budget_ms - elapsed_ms,
                )
                estimated_tick_ms = max(self._estimated_training_tick_ms, 0.05)
                chunk = max(1, int(budget_left / estimated_tick_ms))
                chunk = min(chunk, self.MAX_INTERACTIVE_BATCH_TICKS, remaining)
            else:
                chunk = remaining

            chunk_started = perf_counter()
            results = self.session.step_many(
                chunk,
                stop_after_generation=max_generations is not None,
            )
            chunk_elapsed_ms = max(
                (perf_counter() - chunk_started) * 1_000.0,
                1e-9,
            )
            completed = len(results)
            if completed <= 0:
                break
            sample_tick_ms = chunk_elapsed_ms / completed
            self._estimated_training_tick_ms = (
                sample_tick_ms
                if self._estimated_training_tick_ms <= 0.0
                else self._estimated_training_tick_ms * 0.70 + sample_tick_ms * 0.30
            )
            self.training_steps += completed
            advanced += completed

            if (
                self.render_enabled
                and advanced < self.steps_per_frame
                and (perf_counter() - started_at) * 1_000.0
                >= self.effective_training_frame_budget_ms
            ):
                self._training_slice_capped = True
                break

            if (
                max_generations is not None
                and self.session.completed_generations - starting_generations
                >= max_generations
            ):
                self.running = False
                break

        self._last_training_slice_steps = advanced
        elapsed = max(perf_counter() - started_at, 1e-12)
        decisions = self.session.environment_decisions - starting_decisions
        self._training_slice_ms = elapsed * 1_000.0
        tick_sample = advanced / elapsed
        decision_sample = decisions / elapsed
        smoothing = 0.20
        self._training_ticks_per_second = (
            tick_sample
            if self._training_ticks_per_second <= 0.0
            else self._training_ticks_per_second * (1.0 - smoothing)
            + tick_sample * smoothing
        )
        self._environment_decisions_per_second = (
            decision_sample
            if self._environment_decisions_per_second <= 0.0
            else self._environment_decisions_per_second * (1.0 - smoothing)
            + decision_sample * smoothing
        )
        return advanced

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
                "show_sensor_rays": self.show_sensor_rays,
                "show_population_cars": False,
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
            self._last_training_slice_steps = 0
            self._training_slice_capped = False
            self.handle_events()
            if self.race is not None:
                # A race is intentionally real-time and never inherits the
                # accelerated training multiplier.
                if not self.race.finished:
                    self.race.step(self.controls_from_keyboard())
            elif not self.paused:
                advanced = self._advance_training_slice(
                    starting_generations=starting_generations,
                    max_training_steps=max_training_steps,
                    max_generations=max_generations,
                )

                if self.show_population_cars and advanced:
                    manager = self._ensure_population_rollouts()
                    manager.step()

            if self.render_enabled:
                self.draw()
                pygame.display.flip()
                elapsed_ms = clock.tick(fps)
                if elapsed_ms > 0:
                    self._frame_time_ms = float(elapsed_ms)
                    sample_fps = 1_000.0 / elapsed_ms
                    self._render_fps = (
                        sample_fps
                        if self._render_fps <= 0.0
                        else self._render_fps * 0.85 + sample_fps * 0.15
                    )
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
        try:
            self.session.close()
        finally:
            if self.render_enabled:
                pygame.display.quit()


__all__ = ("DrivingLearningGame",)
