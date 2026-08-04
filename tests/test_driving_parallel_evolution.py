"""Concurrent population-evaluation guarantees for the Driving Lab."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drivingGameRL.src.environment import DrivingEnv, StepResult
from drivingGameRL.src.ml import DQNConfig
from drivingGameRL.src.ml.evolution import EvolutionConfig, PopulationTrainer


def _evolution(**changes) -> EvolutionConfig:
    values = {
        "algorithm": "genetic_dqn",
        "population_size": 4,
        "elite_count": 1,
        "tournament_size": 2,
        "evaluation_steps": 3,
        "mutation_rate": 0.0,
        "mutation_std": 0.0,
        "seed": 101,
    }
    values.update(changes)
    return EvolutionConfig(**values)


def _dqn(seed: int = 101) -> DQNConfig:
    return DQNConfig(
        algorithm="double_dqn",
        hidden_sizes=(8,),
        replay_capacity=32,
        batch_size=1,
        warmup_steps=1,
        target_sync_interval=2,
        epsilon_start=0.25,
        epsilon_end=0.25,
        seed=seed,
    )


def _parameters(trainer: PopulationTrainer) -> tuple[tuple[torch.Tensor, ...], ...]:
    return tuple(
        tuple(
            parameter.detach().cpu().clone()
            for parameter in member.agent.online_network.parameters()
        )
        for member in trainer.population
    )


def _assert_nested_equal(
    case: unittest.TestCase,
    first: object,
    second: object,
    *,
    path: str = "state",
) -> None:
    """Compare checkpoint structures without weakening exact tensor equality."""

    case.assertIs(type(first), type(second), path)
    if isinstance(first, torch.Tensor):
        case.assertTrue(torch.equal(first, second), path)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second, err_msg=path)
    elif isinstance(first, Mapping):
        case.assertEqual(tuple(first), tuple(second), path)
        for key in first:
            _assert_nested_equal(
                case, first[key], second[key], path=f"{path}.{key}"
            )
    elif isinstance(first, (list, tuple)):
        case.assertEqual(len(first), len(second), path)
        for index, (first_item, second_item) in enumerate(zip(first, second)):
            _assert_nested_equal(
                case,
                first_item,
                second_item,
                path=f"{path}[{index}]",
            )
    else:
        case.assertEqual(first, second, path)


class _ConcurrencyProbe:
    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties)
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.thread_ids: set[int] = set()

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.thread_ids.add(threading.get_ident())

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _ProbedEnv(DrivingEnv):
    def __init__(self, seed: int, probe: _ConcurrencyProbe, rank: int) -> None:
        self._probe = probe
        self._rank = rank
        super().__init__(
            "harbor_loop",
            seed=seed,
            max_steps=3,
            random_start_curriculum=True,
        )

    def step(self, action):
        self._probe.enter()
        try:
            self._probe.barrier.wait(timeout=3.0)
            # Reverse completion order relative to population order.
            time.sleep((3 - self._rank) * 0.002)
            return super().step(action)
        finally:
            self._probe.leave()


class DrivingParallelEvolutionTests(unittest.TestCase):
    def _track(self, trainer: PopulationTrainer) -> PopulationTrainer:
        self.addCleanup(trainer.close)
        return trainer

    def test_direct_population_default_uses_lifetime_scaled_exploration(self):
        trainer = self._track(
            PopulationTrainer(
                _evolution(evaluation_steps=321),
                parallel_workers=1,
            )
        )

        self.assertEqual(trainer.dqn_config.epsilon_start, 0.30)
        self.assertEqual(trainer.dqn_config.epsilon_end, 0.05)
        self.assertEqual(trainer.dqn_config.epsilon_decay_steps, 321)
        self.assertEqual(trainer.dqn_config.warmup_steps, 96)
        self.assertEqual(trainer.dqn_config.train_interval, 4)

    def test_legacy_high_epsilon_population_checkpoint_remains_loadable(self):
        config = _evolution(evaluation_steps=321)
        legacy = self._track(
            PopulationTrainer(
                config,
                DQNConfig(seed=config.seed),
                parallel_workers=1,
            )
        )
        restored = self._track(
            PopulationTrainer(
                config,
                parallel_workers=1,
            )
        )

        restored.load_state_dict(deepcopy(legacy.state_dict()))

        self.assertEqual(legacy.dqn_config.epsilon_start, 1.0)
        self.assertEqual(restored.dqn_config.epsilon_start, 0.30)
        self.assertEqual(restored.dqn_config.epsilon_decay_steps, 321)
        for legacy_member, restored_member in zip(
            legacy.population,
            restored.population,
        ):
            for legacy_parameter, restored_parameter in zip(
                legacy_member.agent.online_network.parameters(),
                restored_member.agent.online_network.parameters(),
            ):
                self.assertTrue(torch.equal(legacy_parameter, restored_parameter))

    def test_worker_resolution_is_bounded_and_explicit_workers_can_oversubscribe(self):
        config = _evolution(population_size=4)
        with patch("drivingGameRL.src.ml.evolution.os.cpu_count", return_value=1):
            automatic = self._track(
                PopulationTrainer(config, _dqn(), parallel_workers=None)
            )
            explicit = self._track(
                PopulationTrainer(config, _dqn(), parallel_workers=4)
            )

        self.assertEqual(automatic.parallel_workers, 1)
        self.assertEqual(explicit.parallel_workers, 4)
        self.assertIsNone(automatic.requested_parallel_workers)
        with self.assertRaisesRegex(ValueError, "positive integer or None"):
            PopulationTrainer(config, _dqn(), parallel_workers=0)
        with self.assertRaisesRegex(ValueError, "positive integer or None"):
            PopulationTrainer(config, _dqn(), parallel_workers=True)

    def test_all_members_have_isolated_envs_and_the_same_generation_spawn(self):
        seeds: list[int] = []

        def factory(seed: int) -> DrivingEnv:
            seeds.append(seed)
            return DrivingEnv(
                "harbor_loop",
                seed=seed,
                max_steps=3,
                random_start_curriculum=True,
            )

        trainer = self._track(
            PopulationTrainer(
                _evolution(),
                _dqn(),
                env_factory=factory,
                parallel_workers=4,
                auto_evolve=False,
            )
        )

        self.assertEqual(len(seeds), 4)
        self.assertEqual(len(set(seeds)), 1)
        self.assertEqual(len({id(env) for env in trainer.member_environments}), 4)
        positions = {env.vehicle.state.position for env in trainer.member_environments}
        self.assertEqual(len(positions), 1)
        self.assertEqual(len(set(trainer.member_observations)), 1)

    def test_custom_environment_contract_requires_a_factory_and_equal_resets(self):
        class CustomDrivingEnv(DrivingEnv):
            pass

        with self.assertRaisesRegex(TypeError, "use env_factory"):
            PopulationTrainer(
                _evolution(),
                _dqn(),
                env=CustomDrivingEnv("harbor_loop", seed=101),
            )

        reset_counts: list[list[int]] = []

        def factory(seed: int) -> DrivingEnv:
            env = CustomDrivingEnv(
                "harbor_loop",
                seed=seed,
                max_steps=3,
                random_start_curriculum=True,
            )
            count = [0]
            original_reset = env.reset

            def counted_reset(*args, **kwargs):
                count[0] += 1
                return original_reset(*args, **kwargs)

            env.reset = counted_reset
            reset_counts.append(count)
            return env

        trainer = self._track(
            PopulationTrainer(
                _evolution(),
                _dqn(),
                env_factory=factory,
                parallel_workers=4,
            )
        )
        self.assertEqual([count[0] for count in reset_counts], [1, 1, 1, 1])
        self.assertEqual(len({type(env) for env in trainer.member_environments}), 1)

    def test_active_members_really_overlap_in_the_bounded_thread_pool(self):
        probe = _ConcurrencyProbe(parties=4)
        created = 0

        def factory(seed: int) -> DrivingEnv:
            nonlocal created
            env = _ProbedEnv(seed, probe, created)
            created += 1
            return env

        trainer = self._track(
            PopulationTrainer(
                _evolution(algorithm="genetic"),
                _dqn(),
                env_factory=factory,
                parallel_workers=4,
                auto_evolve=False,
            )
        )

        step = trainer.step()

        self.assertEqual(probe.peak, 4)
        self.assertEqual(len(probe.thread_ids), 4)
        self.assertEqual(step.active_member_indices, (0, 1, 2, 3))
        self.assertEqual([env.steps for env in trainer.member_environments], [1] * 4)
        self.assertEqual(trainer.telemetry()["last_tick_member_count"], 4)

    def test_parallel_completion_timing_does_not_change_results_or_weights(self):
        config = _evolution()
        sequential = self._track(
            PopulationTrainer(
                config,
                _dqn(),
                parallel_workers=1,
                auto_evolve=False,
            )
        )
        created = 0

        def delayed_factory(seed: int) -> DrivingEnv:
            nonlocal created
            rank = created
            created += 1
            env = DrivingEnv(
                "harbor_loop",
                seed=seed,
                max_steps=3,
                random_start_curriculum=True,
            )
            original_step = env.step

            def delayed_step(action):
                time.sleep((3 - rank) * 0.001)
                return original_step(action)

            env.step = delayed_step
            return env

        parallel = self._track(
            PopulationTrainer(
                config,
                _dqn(),
                env_factory=delayed_factory,
                parallel_workers=4,
                auto_evolve=False,
            )
        )

        for _ in range(config.evaluation_steps):
            sequential.step()
            parallel.step()

        self.assertEqual(
            [member.result for member in parallel.population],
            [member.result for member in sequential.population],
        )
        for parallel_member, sequential_member in zip(
            _parameters(parallel), _parameters(sequential)
        ):
            for parallel_parameter, sequential_parameter in zip(
                parallel_member, sequential_member
            ):
                self.assertTrue(torch.equal(parallel_parameter, sequential_parameter))

    def test_multiple_evolved_generations_are_bit_exact_across_worker_counts(self):
        config = _evolution(evaluation_steps=2, mutation_rate=0.2, mutation_std=0.01)
        sequential = self._track(
            PopulationTrainer(config, _dqn(), parallel_workers=1)
        )
        parallel = self._track(
            PopulationTrainer(config, _dqn(), parallel_workers=4)
        )

        for _ in range(config.evaluation_steps * 3):
            sequential.step()
            parallel.step()

        self.assertEqual(sequential.generation, 3)
        self.assertEqual(parallel.generation, 3)
        _assert_nested_equal(self, sequential.state_dict(), parallel.state_dict())

    def test_chunked_steps_are_bit_exact_to_repeated_single_ticks(self):
        config = _evolution(evaluation_steps=5, mutation_rate=0.2, mutation_std=0.01)
        repeated = self._track(
            PopulationTrainer(config, _dqn(), parallel_workers=4)
        )
        chunked = self._track(
            PopulationTrainer(config, _dqn(), parallel_workers=4)
        )

        repeated_steps = tuple(repeated.step() for _ in range(8))
        chunked_steps = chunked.step_many(8)

        self.assertEqual(len(chunked_steps), 8)
        self.assertEqual(
            tuple(
                (
                    step.generation,
                    step.member_id,
                    step.action,
                    step.reward,
                    step.generation_completed,
                    step.evolved,
                )
                for step in chunked_steps
            ),
            tuple(
                (
                    step.generation,
                    step.member_id,
                    step.action,
                    step.reward,
                    step.generation_completed,
                    step.evolved,
                )
                for step in repeated_steps
            ),
        )
        _assert_nested_equal(self, repeated.state_dict(), chunked.state_dict())
        telemetry = chunked.telemetry()
        self.assertEqual(telemetry["last_batch_ticks"], 8)
        self.assertEqual(telemetry["last_batch_decisions"], 32)
        self.assertGreater(telemetry["last_batch_ms"], 0.0)
        self.assertGreater(telemetry["decision_throughput"], 0.0)

    def test_chunk_submits_once_per_member_and_preserves_real_concurrency(self):
        probe = _ConcurrencyProbe(parties=4)
        created = 0

        def factory(seed: int) -> DrivingEnv:
            nonlocal created
            env = _ProbedEnv(seed, probe, created)
            created += 1
            return env

        trainer = self._track(
            PopulationTrainer(
                _evolution(algorithm="genetic", evaluation_steps=3),
                _dqn(),
                env_factory=factory,
                parallel_workers=4,
                auto_evolve=False,
            )
        )

        steps = trainer.step_many(3)

        self.assertEqual(len(steps), 3)
        self.assertEqual(probe.peak, 4)
        self.assertEqual(len(probe.thread_ids), 4)
        self.assertEqual([env.steps for env in trainer.member_environments], [3] * 4)
        self.assertTrue(steps[-1].generation_completed)

    def test_chunk_can_stop_exactly_at_a_generation_boundary(self):
        trainer = self._track(
            PopulationTrainer(
                _evolution(algorithm="genetic", evaluation_steps=3),
                _dqn(),
                parallel_workers=4,
            )
        )

        steps = trainer.step_many(20, stop_after_generation=True)

        self.assertEqual(len(steps), 3)
        self.assertTrue(steps[-1].evolved)
        self.assertEqual(trainer.generation, 1)
        self.assertEqual(trainer.environment_decisions, 12)

    def test_chunk_size_requires_a_positive_integer(self):
        trainer = self._track(
            PopulationTrainer(_evolution(), _dqn(), parallel_workers=1)
        )
        for invalid in (True, 0, -1, 2.5):
            with self.subTest(max_ticks=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    trainer.step_many(invalid)

    def test_generation_evolves_only_after_every_member_finishes(self):
        trainer = self._track(
            PopulationTrainer(
                _evolution(algorithm="genetic", evaluation_steps=2),
                _dqn(),
                parallel_workers=4,
                auto_evolve=True,
            )
        )

        first = trainer.step()
        second = trainer.step()

        self.assertFalse(first.generation_completed)
        self.assertEqual(first.member_results, ())
        self.assertTrue(second.generation_completed)
        self.assertTrue(second.evolved)
        self.assertEqual(len(second.member_results), 4)
        self.assertEqual(second.active_member_indices, (0, 1, 2, 3))
        self.assertEqual(trainer.generation, 1)
        self.assertEqual(len(trainer.history), 1)
        self.assertEqual(trainer.telemetry()["environment_decisions"], 8)
        self.assertEqual(trainer.active_member_indices, (0, 1, 2, 3))

    def test_early_member_completion_is_reported_and_removed_from_active_set(self):
        trainer = self._track(
            PopulationTrainer(
                _evolution(algorithm="genetic"),
                _dqn(),
                parallel_workers=4,
                auto_evolve=False,
            )
        )
        early_env = trainer.member_environments[2]
        observation = trainer.member_observations[2]

        def finish_early(_action):
            return StepResult(
                observation=observation,
                reward=7.0,
                terminated=True,
                truncated=False,
                info={"laps": 0, "progress": 0.2},
            )

        early_env.step = finish_early
        tick = trainer.step()

        self.assertTrue(tick.member_completed)
        self.assertEqual(tick.member_index, 2)
        self.assertEqual(tick.result.member_id, trainer.population[2].member_id)
        self.assertEqual(len(tick.member_results), 1)
        self.assertEqual(trainer.active_member_indices, (0, 1, 3))

    def test_curriculum_unlock_is_merged_only_at_generation_boundary(self):
        trainer = self._track(
            PopulationTrainer(
                _evolution(algorithm="genetic", evaluation_steps=1),
                _dqn(),
                parallel_workers=4,
                auto_evolve=True,
            )
        )
        generation_ready_seen: list[bool] = []
        for index, env in enumerate(trainer.member_environments):
            observation = trainer.member_observations[index]

            def terminal_step(_action, *, item=index, obs=observation):
                generation_ready_seen.append(trainer._generation_curriculum_ready)
                return StepResult(
                    observation=obs,
                    reward=20.0 if item == 0 else 0.0,
                    terminated=True,
                    truncated=False,
                    info={
                        "curriculum_lap_completed": item == 0,
                        "laps": 1 if item == 0 else 0,
                        "episode_lap_progress": 1.0 if item == 0 else 0.0,
                    },
                )

            env.step = terminal_step

        tick = trainer.step()

        self.assertEqual(generation_ready_seen, [False] * 4)
        self.assertTrue(tick.evolved)
        self.assertTrue(trainer._generation_curriculum_ready)
        self.assertFalse(trainer._pending_curriculum_unlock)
        self.assertTrue(
            all(env.curriculum_ready for env in trainer.member_environments)
        )

    def test_v1_checkpoint_restarts_unfinished_members_with_any_worker_count(self):
        config = _evolution(algorithm="genetic")
        source = self._track(
            PopulationTrainer(
                config,
                _dqn(),
                parallel_workers=4,
                auto_evolve=False,
            )
        )
        source.step()
        legacy_state = deepcopy(source.state_dict())
        legacy_state.pop("environment_decisions")

        restored = self._track(
            PopulationTrainer(
                config,
                _dqn(),
                parallel_workers=1,
                auto_evolve=False,
            )
        )
        restored.load_state_dict(legacy_state)

        self.assertEqual(restored.parallel_workers, 1)
        self.assertEqual(restored.active_member_indices, (0, 1, 2, 3))
        self.assertEqual([env.steps for env in restored.member_environments], [0] * 4)
        self.assertNotIn("parallel_workers", restored.state_dict()["evolution_config"])

    def test_hybrid_agents_and_replay_buffers_remain_isolated(self):
        trainer = self._track(
            PopulationTrainer(
                _evolution(),
                _dqn(),
                parallel_workers=4,
                auto_evolve=False,
            )
        )

        trainer.step()

        self.assertEqual(
            len({id(member.agent.replay) for member in trainer.population}), 4
        )
        self.assertTrue(
            all(len(member.agent.replay) == 1 for member in trainer.population)
        )
        self.assertTrue(
            all(member.agent.gradient_steps == 1 for member in trainer.population)
        )
        first_parameters = [
            next(member.agent.online_network.parameters())
            for member in trainer.population
        ]
        self.assertEqual(
            len(
                {
                    parameter.untyped_storage().data_ptr()
                    for parameter in first_parameters
                }
            ),
            4,
        )

    def test_close_is_idempotent_and_prevents_more_steps(self):
        trainer = PopulationTrainer(
            _evolution(algorithm="genetic"),
            _dqn(),
            parallel_workers=4,
            auto_evolve=False,
        )
        trainer.step()
        self.assertIsNotNone(trainer._executor)

        trainer.close()
        trainer.close()

        self.assertIsNone(trainer._executor)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            trainer.step()

    def test_worker_failure_makes_a_partially_executed_tick_fail_stop(self):
        trainer = self._track(
            PopulationTrainer(
                _evolution(algorithm="genetic"),
                _dqn(),
                parallel_workers=4,
                auto_evolve=False,
            )
        )
        barrier = threading.Barrier(4)
        for index, env in enumerate(trainer.member_environments):
            original_step = env.step

            def synchronized_step(action, *, member=index, advance=original_step):
                barrier.wait(timeout=3.0)
                if member == 0:
                    raise ValueError("member zero failed")
                return advance(action)

            env.step = synchronized_step

        with self.assertRaisesRegex(RuntimeError, "evaluation failed") as raised:
            trainer.step()

        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertEqual([env.steps for env in trainer.member_environments], [0, 1, 1, 1])
        self.assertEqual([row["evaluation_step"] for row in trainer.member_runtime], [0] * 4)
        self.assertTrue(trainer.telemetry()["worker_failed"])
        self.assertEqual(trainer.telemetry()["worker_failure_type"], "ValueError")
        self.assertIsNone(trainer._executor)
        for operation in (
            trainer.step,
            trainer.reset,
            trainer.evolve,
            trainer.state_dict,
        ):
            with self.assertRaisesRegex(RuntimeError, "worker failure"):
                operation()


if __name__ == "__main__":
    unittest.main()
