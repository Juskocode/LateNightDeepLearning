"""Playable Pygame view over the deterministic driving environment."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pygame

from .circuits import circuit_names
from .environment import DrivingEnv, StepResult
from .math2d import clamp, wrap_angle
from .rendering import (
    CircuitRenderer,
    RacingGhostRenderer,
    TelemetryHUD,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    draw_sensor_rays,
)
from .sprites import CarSprite, ParticleSystem
from .vehicle import CarBuild, DriverControls, MAX_UPGRADE_LEVEL


class DrivingGame:
    def __init__(
        self,
        circuit: str = "harbor_loop",
        *,
        build: CarBuild | None = None,
        seed: int | None = None,
        render: bool = True,
        car_sprite_path: str | Path | None = None,
    ):
        pygame.init()
        pygame.font.init()
        self.seed = seed
        self.render_enabled = render
        if render:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Late Night Driving Lab")
        else:
            self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.env = DrivingEnv(circuit, build=build, seed=seed)
        self.circuit_renderer = CircuitRenderer()
        self.ghost_renderer = RacingGhostRenderer()
        self.hud = TelemetryHUD()
        self.particles = ParticleSystem(seed)
        self.car = CarSprite(self.env.vehicle.build, car_sprite_path)
        self.car_group = pygame.sprite.GroupSingle(self.car)
        self.last_controls = DriverControls()
        self.show_sensors = True
        self.show_ghost = True
        self.running = True
        self.car.sync(self.env.vehicle)

    def reset(self) -> None:
        self.env.reset(seed=self.seed)
        self.particles.reset()
        self.last_controls = DriverControls()
        self.car.sync(self.env.vehicle)

    def step(self, controls: DriverControls) -> StepResult:
        self.last_controls = controls.clamped()
        active_terrain = self.env.circuit.terrain_at(self.env.vehicle.state.position)
        result = self.env.step_controls(self.last_controls)
        self.particles.emit_drive(
            self.env.vehicle, active_terrain, self.last_controls, self.env.fixed_dt
        )
        impact = float(result.info["impact_speed"])
        if bool(result.info["collision_started"]):
            self.particles.emit_collision(self.env.vehicle.state.position, impact)
        self.particles.update(self.env.fixed_dt)
        self.car.sync(self.env.vehicle)
        return result

    def telemetry(self) -> dict[str, object]:
        snapshot = self.env.telemetry()
        snapshot["particles"] = len(self.particles)
        snapshot["sprite_asset_loaded"] = self.car.using_external_image
        snapshot["ghost_enabled"] = self.show_ghost
        return snapshot

    def draw(self) -> pygame.Surface:
        self.screen.blit(self.circuit_renderer.surface_for(self.env.circuit), (0, 0))
        if self.show_ghost:
            self.ghost_renderer.draw(
                self.screen,
                self.env.ghost_pose_at(),
                self.env.best_lap_trajectory,
            )
        self.particles.draw(self.screen)
        if self.show_sensors:
            draw_sensor_rays(self.screen, self.env)
        self.car_group.draw(self.screen)
        self.hud.draw(
            self.screen,
            self.env,
            self.last_controls,
            len(self.particles),
            ghost_enabled=self.show_ghost,
        )
        return self.screen

    def save_screenshot(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.draw()
        pygame.image.save(self.screen, str(output))
        return output

    def cycle_upgrade(self, component: str) -> CarBuild:
        build = self.env.vehicle.build
        next_level = (getattr(build, component) + 1) % (MAX_UPGRADE_LEVEL + 1)
        build = replace(build, **{component: next_level})
        self.env.vehicle.set_build(build)
        self.env.restart_lap_candidate(wait_for_start=True)
        self.car.set_build(build)
        return build

    def cycle_circuit(self) -> str:
        names = circuit_names()
        current = names.index(self.env.circuit.slug)
        name = names[(current + 1) % len(names)]
        self.env.change_circuit(name)
        self.particles.reset()
        self.last_controls = DriverControls()
        self.car.sync(self.env.vehicle)
        return name

    def controls_from_keyboard(self) -> DriverControls:
        keys = pygame.key.get_pressed()
        throttle = float(keys[pygame.K_w] or keys[pygame.K_UP])
        throttle -= float(keys[pygame.K_s] or keys[pygame.K_DOWN]) * 0.65
        steering = float(keys[pygame.K_d] or keys[pygame.K_RIGHT])
        steering -= float(keys[pygame.K_a] or keys[pygame.K_LEFT])
        brake = float(keys[pygame.K_SPACE])
        return DriverControls(throttle=throttle, steering=steering, brake=brake)

    def autopilot_controls(self) -> DriverControls:
        projection = self.env.circuit.project(self.env.vehicle.state.position)
        speed_ratio = self.env.vehicle.state.speed / self.env.vehicle.build.max_speed
        state = self.env.vehicle.state

        # Recover deliberately instead of repeatedly applying throttle into a
        # barrier. Reverse is useful when the nose still points away from the
        # center line; otherwise a gentle inward input resumes the lap.
        boundary_ratio = (
            abs(projection.signed_offset) / self.env.circuit.collision_radius
        )
        if boundary_ratio > 0.88:
            inward = projection.point - state.position
            inward_direction = math.atan2(inward.y, inward.x)
            forward_alignment = math.cos(inward_direction - state.heading)
            if forward_alignment < 0.12:
                desired_body_heading = wrap_angle(inward_direction - math.pi)
                reverse_error = wrap_angle(desired_body_heading - state.heading)
                return DriverControls(
                    throttle=-0.58,
                    steering=clamp(-reverse_error * 1.9, -1.0, 1.0),
                )
            recovery_error = wrap_angle(inward_direction - state.heading)
            return DriverControls(
                throttle=0.48,
                steering=clamp(recovery_error * 1.9, -1.0, 1.0),
            )

        lookahead_distance = 28.0 + 48.0 * min(1.0, speed_ratio)
        lookahead = lookahead_distance / self.env.circuit.length
        target, tangent = self.env.circuit.point_tangent_at(
            projection.progress + lookahead
        )
        tangent_angle = math.atan2(projection.tangent.y, projection.tangent.x)
        corner_angle = 0.0
        for preview_distance in (90.0, 155.0, 230.0):
            _, future_tangent = self.env.circuit.point_tangent_at(
                projection.progress + preview_distance / self.env.circuit.length
            )
            future_angle = math.atan2(future_tangent.y, future_tangent.x)
            corner_angle = max(
                corner_angle, abs(wrap_angle(future_angle - tangent_angle))
            )
        target_speed_ratio = clamp(0.70 - 0.68 * corner_angle / math.pi, 0.18, 0.70)

        to_target = target - state.position
        desired = (
            math.atan2(to_target.y, to_target.x)
            if to_target.length() > 2.0
            else math.atan2(tangent.y, tangent.x)
        )
        error = wrap_angle(desired - state.heading)
        steering = clamp(error * 2.25, -1.0, 1.0)
        corner = min(1.0, abs(error) / 1.15)
        too_fast = speed_ratio > target_speed_ratio + 0.035
        throttle = 0.0 if too_fast else 1.0 - corner * 0.55
        brake = (
            min(0.82, 0.35 + (speed_ratio - target_speed_ratio) * 2.5)
            if too_fast
            else 0.0
        )
        return DriverControls(throttle=throttle, steering=steering, brake=brake)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_c:
                    self.cycle_circuit()
                elif event.key == pygame.K_v:
                    self.show_sensors = not self.show_sensors
                elif event.key == pygame.K_g:
                    self.show_ghost = not self.show_ghost
                elif event.key == pygame.K_1:
                    self.cycle_upgrade("motor")
                elif event.key == pygame.K_2:
                    self.cycle_upgrade("wheels")
                elif event.key == pygame.K_3:
                    self.cycle_upgrade("suspension")
                elif event.key == pygame.K_4:
                    self.cycle_upgrade("grip")
                elif event.key == pygame.K_F12:
                    self.save_screenshot("driving-screenshot.png")

    def run(
        self, *, fps: int = 60, max_steps: int | None = None, autopilot: bool = False
    ) -> None:
        clock = pygame.time.Clock()
        completed = 0
        self.running = True
        while self.running and (max_steps is None or completed < max_steps):
            self.handle_events()
            controls = (
                self.autopilot_controls()
                if autopilot
                else self.controls_from_keyboard()
            )
            self.step(controls)
            self.draw()
            if self.render_enabled:
                pygame.display.flip()
                clock.tick(fps)
            completed += 1

    def close(self) -> None:
        if self.render_enabled:
            pygame.display.quit()
