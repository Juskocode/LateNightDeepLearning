"""Safety-prior behavior and learning-runtime integration tests."""

from __future__ import annotations

import unittest

import numpy as np

from drivingGameRL.src.environment import DrivingAction, DrivingEnv, StepResult
from drivingGameRL.src.learning_runtime import (
    ChampionRace,
    DrivingLearningSession,
    LearningRuntimeConfig,
)
from drivingGameRL.src.ml import DQNConfig
from drivingGameRL.src.ml.evolution import EvolutionConfig, PopulationTrainer
from drivingGameRL.src.sensor_clearance import (
    SensorClearancePolicy,
    SensorClearanceStats,
)
from drivingGameRL.src.vehicle import DriverControls


def _observation(
    *,
    speed: float = 0.1,
    rays: tuple[float, ...] = (0.9,) * 9,
) -> tuple[float, ...]:
    if len(rays) != 9:
        raise ValueError("test observations require nine rays")
    return (
        speed,
        speed,
        0.0,
        0.0,
        0.0,
        1.0,
        0.25,
        *rays,
    )


def _result(observation: tuple[float, ...], reward: float = 0.25) -> StepResult:
    return StepResult(
        observation=observation,
        reward=reward,
        terminated=False,
        truncated=False,
        info={"laps": 0, "episode_lap_progress": 0.1},
    )


def _tiny_dqn(seed: int = 11) -> DQNConfig:
    return DQNConfig(
        hidden_sizes=(8,),
        replay_capacity=32,
        batch_size=2,
        warmup_steps=32,
        epsilon_start=0.0,
        epsilon_end=0.0,
        seed=seed,
    )


class SensorClearancePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SensorClearancePolicy()

    def test_clear_forward_fan_leaves_every_proposed_action_untouched(self):
        observation = _observation(rays=(0.7, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.7))

        for action in DrivingAction:
            with self.subTest(action=action):
                decision = self.policy.decide(observation, action)
                self.assertEqual(decision.proposed_action, int(action))
                self.assertEqual(decision.executed_action, int(action))
                self.assertFalse(decision.intervened)
                self.assertFalse(decision.dangerous)
                self.assertEqual(decision.reason, "clear_road")

    def test_critical_high_speed_corridor_brakes_before_impact(self):
        observation = _observation(
            speed=0.75,
            rays=(0.5, 0.5, 0.6, 0.07, 0.05, 0.08, 0.8, 0.9, 1.0),
        )

        decision = self.policy.decide(observation, DrivingAction.ACCELERATE)

        self.assertEqual(decision.proposed_action, int(DrivingAction.ACCELERATE))
        self.assertEqual(decision.executed_action, int(DrivingAction.BRAKE))
        self.assertTrue(decision.intervened)
        self.assertTrue(decision.dangerous)
        self.assertEqual(decision.reason, "critical_brake")

    def test_speed_lookahead_intervenes_earlier_without_changing_clear_road(self):
        rays = (0.9, 0.9, 0.8, 0.62, 0.53, 0.72, 0.9, 0.9, 0.9)

        slow = self.policy.decide(
            _observation(speed=0.10, rays=rays),
            DrivingAction.ACCELERATE,
        )
        fast = self.policy.decide(
            _observation(speed=0.90, rays=rays),
            DrivingAction.ACCELERATE,
        )

        self.assertFalse(slow.dangerous)
        self.assertEqual(slow.executed_action, int(DrivingAction.ACCELERATE))
        self.assertTrue(fast.dangerous)
        self.assertGreater(fast.danger_threshold, slow.danger_threshold)
        self.assertIn(
            fast.executed_action,
            (int(DrivingAction.STEER_LEFT), int(DrivingAction.STEER_RIGHT)),
        )

    def test_projected_offset_biases_equal_green_space_toward_track_center(self):
        rays = (0.7, 0.7, 0.7, 0.7, 0.30, 0.7, 0.7, 0.7, 0.7)
        positive_offset = list(_observation(speed=0.20, rays=rays))
        positive_offset[3] = 0.35
        positive_offset[4] = 0.40
        negative_offset = list(_observation(speed=0.20, rays=rays))
        negative_offset[3] = -0.35
        negative_offset[4] = -0.40

        left = self.policy.decide(
            positive_offset,
            DrivingAction.ACCELERATE,
        )
        right = self.policy.decide(
            negative_offset,
            DrivingAction.ACCELERATE,
        )

        self.assertEqual(left.executed_action, int(DrivingAction.STEER_LEFT))
        self.assertGreater(left.left_utility, left.right_utility)
        self.assertEqual(right.executed_action, int(DrivingAction.STEER_RIGHT))
        self.assertGreater(right.right_utility, right.left_utility)

    def test_slow_car_steers_toward_the_weighted_more_open_side(self):
        left_open = _observation(
            speed=0.12,
            rays=(1.0, 0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1),
        )
        right_open = _observation(
            speed=0.12,
            rays=(0.1, 0.1, 0.1, 0.1, 0.2, 0.7, 0.8, 0.9, 1.0),
        )

        left = self.policy.decide(left_open, DrivingAction.ACCELERATE)
        right = self.policy.decide(right_open, DrivingAction.ACCELERATE)

        self.assertEqual(left.executed_action, int(DrivingAction.STEER_LEFT))
        self.assertGreater(left.left_open_space, left.right_open_space)
        self.assertEqual(right.executed_action, int(DrivingAction.STEER_RIGHT))
        self.assertGreater(right.right_open_space, right.left_open_space)
        self.assertEqual(left.ray_clearances, left_open[-9:])
        self.assertEqual(right.ray_clearances, right_open[-9:])

    def test_green_rays_receive_an_explicit_corridor_utility_bonus(self):
        observation = _observation(
            speed=0.20,
            rays=(0.56, 0.56, 0.56, 0.56, 0.30, 0.54, 0.54, 0.54, 0.54),
        )

        decision = self.policy.decide(observation, DrivingAction.ACCELERATE)

        raw_gap = decision.left_open_space - decision.right_open_space
        utility_gap = decision.left_utility - decision.right_utility
        self.assertGreater(utility_gap, raw_gap + 0.10)
        self.assertEqual(decision.executed_action, int(DrivingAction.STEER_LEFT))

    def test_equal_open_space_uses_a_stable_nonrandom_tie_break(self):
        blocked = _observation(speed=0.1, rays=(0.2,) * 9)

        repeated = tuple(
            self.policy.decide(blocked, DrivingAction.ACCELERATE)
            for _ in range(20)
        )
        kept_steer = self.policy.decide(blocked, DrivingAction.STEER_RIGHT)

        self.assertEqual(
            {item.executed_action for item in repeated},
            {int(DrivingAction.STEER_LEFT)},
        )
        self.assertEqual(
            kept_steer.executed_action,
            int(DrivingAction.STEER_RIGHT),
        )
        self.assertFalse(kept_steer.intervened)

    def test_stats_are_ephemeral_transparent_and_bounded(self):
        stats = SensorClearanceStats()
        clear = self.policy.decide(_observation(), DrivingAction.COAST)
        blocked = self.policy.decide(
            _observation(speed=0.8, rays=(0.05,) * 9),
            DrivingAction.ACCELERATE,
        )

        stats.observe(clear)
        stats.observe(blocked)
        snapshot = stats.snapshot()

        self.assertEqual(snapshot["decisions"], 2)
        self.assertEqual(snapshot["interventions"], 1)
        self.assertEqual(snapshot["intervention_rate"], 0.5)
        self.assertEqual(snapshot["proposed_action"], int(DrivingAction.ACCELERATE))
        self.assertEqual(snapshot["executed_action"], int(DrivingAction.BRAKE))
        self.assertEqual(
            snapshot["boundary_threshold"],
            blocked.boundary_threshold,
        )
        self.assertEqual(
            snapshot["boundary_threshold_base"],
            SensorClearancePolicy.BOUNDARY_THRESHOLD,
        )
        self.assertEqual(len(snapshot["ray_clearances"]), 9)

    def test_invalid_action_and_non_nine_ray_observations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            self.policy.decide((0.0,) * 15, DrivingAction.COAST)
        with self.assertRaisesRegex(ValueError, "normalized"):
            self.policy.decide(_observation(rays=(1.2,) * 9), DrivingAction.COAST)
        for action in (True, 99):
            with self.subTest(action=action):
                with self.assertRaisesRegex(ValueError, "Invalid proposed"):
                    self.policy.decide(_observation(), action)


class SensorClearanceIntegrationTests(unittest.TestCase):
    def test_clearance_policy_materially_delays_a_real_wall_collision(self):
        raw_env = DrivingEnv("harbor_loop", seed=1, max_steps=900)
        safe_env = DrivingEnv("harbor_loop", seed=1, max_steps=900)
        policy = SensorClearancePolicy()
        raw_result = None
        safe_result = None

        for _ in range(900):
            raw_result = raw_env.step(DrivingAction.ACCELERATE)
            if raw_result.terminated or raw_result.truncated:
                break
        observation = safe_env.observation()
        for _ in range(900):
            decision = policy.decide(observation, DrivingAction.ACCELERATE)
            safe_result = safe_env.step(decision.executed_action)
            observation = safe_result.observation
            if safe_result.terminated or safe_result.truncated:
                break

        assert raw_result is not None and safe_result is not None
        self.assertGreater(safe_env.steps, raw_env.steps * 2)
        self.assertGreater(
            float(safe_result.info["progress"]),
            float(raw_result.info["progress"]) + 0.40,
        )

    def test_population_rejects_an_incompatible_safety_observation_at_init(self):
        class TwelveFeatureEnv(DrivingEnv):
            def observation(self) -> tuple[float, ...]:
                return super().observation()[:12]

        config = EvolutionConfig(
            algorithm="genetic",
            population_size=2,
            elite_count=1,
            tournament_size=2,
            evaluation_steps=2,
            seed=43,
        )
        dqn = DQNConfig(
            observation_size=12,
            hidden_sizes=(8,),
            replay_capacity=32,
            batch_size=2,
            warmup_steps=32,
            epsilon_start=0.0,
            epsilon_end=0.0,
            seed=43,
        )

        with self.assertRaisesRegex(
            ValueError,
            "sensor-clearance policy requires the exact 16-value",
        ):
            PopulationTrainer(
                config,
                dqn,
                env_factory=lambda seed: TwelveFeatureEnv(
                    "harbor_loop",
                    seed=seed,
                    max_steps=2,
                ),
                parallel_workers=1,
            )

    def test_population_top_level_safety_tracks_the_current_member(self):
        trainer = PopulationTrainer(
            EvolutionConfig(
                algorithm="genetic",
                population_size=2,
                elite_count=1,
                tournament_size=2,
                evaluation_steps=10,
                seed=47,
            ),
            _tiny_dqn(seed=47),
            parallel_workers=1,
        )
        self.addCleanup(trainer.close)
        trainer.population[0].agent.select_action = (
            lambda _state, *, explore=True: int(DrivingAction.COAST)
        )
        trainer.population[1].agent.select_action = (
            lambda _state, *, explore=True: int(DrivingAction.ACCELERATE)
        )
        first_env = trainer.member_environments[0]
        original_step = first_env.step

        def truncate_first(action: int) -> StepResult:
            result = original_step(action)
            return StepResult(
                observation=result.observation,
                reward=result.reward,
                terminated=result.terminated,
                truncated=True,
                info=result.info,
            )

        first_env.step = truncate_first

        trainer.step()
        telemetry = trainer.telemetry()

        self.assertEqual(telemetry["current_member_index"], 1)
        self.assertEqual(
            telemetry["safety_prior"]["proposed_action"],
            telemetry["population"][1]["raw_action"],
        )
        self.assertNotEqual(
            telemetry["safety_prior"]["proposed_action"],
            telemetry["population"][0]["raw_action"],
        )

    def test_standalone_dqn_preserves_proposal_but_replays_executed_action(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="double_dqn",
                evaluation_steps=20,
                population_size=2,
                elite_count=1,
                seed=21,
            ),
            dqn_config=_tiny_dqn(seed=21),
        )
        self.addCleanup(session.close)
        dangerous = _observation(speed=0.8, rays=(0.05,) * 9)
        clear = _observation()
        executed: list[int] = []
        session.observation = dangerous
        session.agent.select_action = lambda _state, explore=True: int(
            DrivingAction.ACCELERATE
        )

        def step(action: int) -> StepResult:
            executed.append(int(action))
            return _result(clear)

        session.env.step = step

        session.step()
        telemetry = session.telemetry()
        transition = list(session.agent.replay)[-1]

        self.assertEqual(executed, [int(DrivingAction.BRAKE)])
        self.assertEqual(transition.action, int(DrivingAction.BRAKE))
        self.assertEqual(telemetry["selected_action"], int(DrivingAction.ACCELERATE))
        self.assertEqual(telemetry["proposed_action"], int(DrivingAction.ACCELERATE))
        self.assertEqual(telemetry["executed_action"], int(DrivingAction.BRAKE))
        self.assertEqual(telemetry["last_action"], int(DrivingAction.BRAKE))
        self.assertTrue(telemetry["safety_intervened"])
        self.assertEqual(telemetry["safety_prior"]["interventions"], 1)

    def test_population_prior_applies_to_genetic_and_hybrid_members(self):
        dangerous = _observation(speed=0.8, rays=(0.05,) * 9)
        clear = _observation()
        for algorithm in ("genetic", "genetic_dqn"):
            with self.subTest(algorithm=algorithm):
                trainer = PopulationTrainer(
                    EvolutionConfig(
                        algorithm=algorithm,
                        population_size=2,
                        elite_count=1,
                        tournament_size=2,
                        evaluation_steps=3,
                        seed=33,
                    ),
                    _tiny_dqn(seed=33),
                    parallel_workers=2,
                    auto_evolve=False,
                )
                try:
                    executed: list[list[int]] = [[], []]
                    for index, (member, runtime) in enumerate(
                        zip(trainer.population, trainer._member_runtimes)
                    ):
                        runtime.observation = np.asarray(dangerous, dtype=np.float32)
                        member.agent.select_action = (
                            lambda _state, explore=True: int(
                                DrivingAction.ACCELERATE
                            )
                        )

                        def step(
                            action: int,
                            *,
                            member_index: int = index,
                        ) -> StepResult:
                            executed[member_index].append(int(action))
                            return _result(clear)

                        runtime.env.step = step
                    trainer._sync_focal_aliases(0)

                    population_step = trainer.step()
                    telemetry = trainer.telemetry()

                    self.assertEqual(
                        executed,
                        [[int(DrivingAction.BRAKE)], [int(DrivingAction.BRAKE)]],
                    )
                    self.assertEqual(
                        population_step.proposed_action,
                        int(DrivingAction.ACCELERATE),
                    )
                    self.assertEqual(
                        population_step.executed_action,
                        int(DrivingAction.BRAKE),
                    )
                    self.assertEqual(population_step.action, int(DrivingAction.BRAKE))
                    self.assertTrue(population_step.safety_intervened)
                    self.assertEqual(telemetry["safety_prior"]["decisions"], 2)
                    self.assertEqual(telemetry["safety_prior"]["interventions"], 2)
                    self.assertTrue(
                        all(row["safety_intervened"] for row in telemetry["population"])
                    )
                    if algorithm == "genetic_dqn":
                        self.assertTrue(
                            all(
                                list(member.agent.replay)[-1].action
                                == int(DrivingAction.BRAKE)
                                for member in trainer.population
                            )
                        )
                finally:
                    trainer.close()

    def test_champion_race_reproduces_the_evaluated_safety_filter(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="double_dqn",
                evaluation_steps=20,
                population_size=2,
                elite_count=1,
                seed=51,
            ),
            dqn_config=_tiny_dqn(seed=51),
        )
        self.addCleanup(session.close)
        race = ChampionRace(session)
        dangerous = _observation(speed=0.8, rays=(0.05,) * 9)
        clear = _observation()
        race.champion_observation = dangerous
        race.agent.q_values = lambda _state: np.asarray(
            [0.0, 5.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        executed: list[int] = []
        race.human_env.step_controls = lambda _controls: _result(clear)

        def champion_step(action: int) -> StepResult:
            executed.append(int(action))
            return _result(clear)

        race.champion_env.step = champion_step

        race.step(DriverControls())
        telemetry = race.telemetry()

        self.assertEqual(executed, [int(DrivingAction.BRAKE)])
        self.assertEqual(
            telemetry["champion_proposed_action"],
            int(DrivingAction.ACCELERATE),
        )
        self.assertEqual(telemetry["champion_action"], int(DrivingAction.BRAKE))
        self.assertTrue(telemetry["champion_safety_intervened"])
        self.assertEqual(telemetry["champion_safety"]["interventions"], 1)

    def test_champion_race_ends_when_only_one_driver_is_collision_truncated(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="double_dqn",
                evaluation_steps=20,
                population_size=2,
                elite_count=1,
                seed=61,
            ),
            dqn_config=_tiny_dqn(seed=61),
        )
        self.addCleanup(session.close)
        race = ChampionRace(session)
        observation = _observation()
        race.champion_observation = observation
        race.agent.q_values = lambda _state: np.asarray(
            [1.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        race.human_env.step_controls = lambda _controls: StepResult(
            observation=observation,
            reward=-1.0,
            terminated=False,
            truncated=True,
            info={"progress": 0.1, "truncation_reason": "collision_loop"},
        )
        race.champion_env.step = lambda _action: StepResult(
            observation=observation,
            reward=0.1,
            terminated=False,
            truncated=False,
            info={"progress": 0.1},
        )

        race.step(DriverControls())

        self.assertTrue(race.finished)
        self.assertEqual(race.winner, "champion")

    def test_safety_observability_does_not_change_checkpoint_schema(self):
        trainer = PopulationTrainer(
            EvolutionConfig(
                algorithm="genetic",
                population_size=2,
                elite_count=1,
                tournament_size=2,
                evaluation_steps=2,
                seed=71,
            ),
            _tiny_dqn(seed=71),
            parallel_workers=1,
        )
        try:
            checkpoint = trainer.state_dict()
        finally:
            trainer.close()

        self.assertNotIn("safety", checkpoint)
        self.assertNotIn("safety_prior", checkpoint)
        self.assertNotIn("clearance_policy", checkpoint)
        self.assertNotIn("safety", checkpoint["population"][0]["agent"])


if __name__ == "__main__":
    unittest.main()
