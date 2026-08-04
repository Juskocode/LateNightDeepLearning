"""Focused health-contract and failure-boundary tests for the Driving Lab."""

from __future__ import annotations

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drivingGameRL.src.learning_health import build_learning_health
from drivingGameRL.src.learning_visualization import _health_alert_label
from drivingGameRL.src.learning_runtime import (
    DrivingLearningSession,
    LearningRuntimeConfig,
)
from drivingGameRL.src.ml import DQNConfig, DrivingDQNAgent
from drivingGameRL.src.ml.evolution import (
    ChampionSnapshot,
    EvaluationResult,
    EvolutionConfig,
    PopulationTrainer,
)


def _dqn(**changes) -> DQNConfig:
    values = {
        "hidden_sizes": (8,),
        "replay_capacity": 32,
        "batch_size": 2,
        "warmup_steps": 2,
        "train_interval": 1,
        "target_sync_interval": 5,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 10,
        "seed": 23,
    }
    values.update(changes)
    return DQNConfig(**values)


def _evolution(**changes) -> EvolutionConfig:
    values = {
        "algorithm": "genetic_dqn",
        "population_size": 2,
        "elite_count": 1,
        "tournament_size": 1,
        "evaluation_steps": 4,
        "mutation_rate": 0.0,
        "mutation_std": 0.0,
        "seed": 23,
    }
    values.update(changes)
    return EvolutionConfig(**values)


class LearningHealthContractTests(unittest.TestCase):
    def test_health_contract_is_finite_stable_and_reports_readiness(self):
        warming = build_learning_health(
            learning={
                "gradient_steps": 0,
                "gradient_norm": 0.0,
                "q_values": [0.2, -0.4],
                "mean_absolute_td_error": 0.0,
            },
            replay={"size": 3, "capacity": 10},
            environment_decisions=3,
            batch_size=4,
            warmup_steps=4,
            gradient_clip=5.0,
        )

        self.assertEqual(warming["status"], "warming_up")
        self.assertTrue(warming["finite"])
        self.assertFalse(warming["replay"]["ready"])
        self.assertEqual(warming["replay"]["readiness_threshold"], 4)
        self.assertAlmostEqual(warming["values"]["q_abs_max"], 0.4)
        self.assertEqual(
            set(warming),
            {
                "status",
                "finite",
                "alerts",
                "replay",
                "optimization",
                "values",
                "safety",
                "throughput",
            },
        )

    def test_malformed_and_nonfinite_diagnostics_degrade_to_critical_zeros(self):
        health = build_learning_health(
            learning={
                "gradient_steps": 2,
                "gradient_norm": math.inf,
                "q_values": [math.nan, 2.0],
                "mean_absolute_td_error": "bad",
            },
            replay={"size": 8, "capacity": 16},
            throughput={"decision_throughput": math.nan},
            environment_decisions=8,
            batch_size=2,
            gradient_clip=5.0,
        )

        self.assertEqual(health["status"], "critical")
        self.assertFalse(health["finite"])
        self.assertEqual(health["optimization"]["gradient_norm"], 0.0)
        self.assertEqual(health["values"]["td_error_abs_mean"], 0.0)
        self.assertEqual(health["throughput"]["decisions_per_second"], 0.0)
        self.assertTrue(any("non_finite" in alert for alert in health["alerts"]))

    def test_non_replay_algorithm_marks_learning_signals_not_applicable(self):
        health = build_learning_health(
            learning={"q_values": [0.1, -0.2]},
            replay={"size": 0, "capacity": 32},
            replay_enabled=False,
        )

        self.assertFalse(health["replay"]["applicable"])
        self.assertFalse(health["optimization"]["applicable"])
        self.assertFalse(health["values"]["td_error_applicable"])
        self.assertTrue(health["values"]["q_applicable"])
        self.assertEqual(health["status"], "healthy")


class DrivingDQNHealthTests(unittest.TestCase):
    def test_agent_reports_replay_updates_gradient_pressure_and_value_scale(self):
        agent = DrivingDQNAgent(_dqn(gradient_clip=1e-8))
        state = np.zeros(16, dtype=np.float32)
        agent.observe(state, 0, 50.0, state, True)
        agent.observe(state, 0, 50.0, state, True)

        telemetry = agent.telemetry(state)
        health = telemetry["health"]

        self.assertTrue(health["replay"]["ready"])
        self.assertEqual(health["optimization"]["updates"], 1)
        self.assertAlmostEqual(health["optimization"]["update_to_decision_ratio"], 0.5)
        self.assertEqual(health["optimization"]["clip_ratio"], 1.0)
        self.assertGreater(health["optimization"]["current_norm_ratio"], 1.0)
        self.assertEqual(agent.gradient_clip_events, 1)
        self.assertTrue(math.isfinite(health["values"]["q_abs_max"]))
        self.assertTrue(math.isfinite(health["values"]["td_error_abs_mean"]))
        self.assertEqual(health["status"], "healthy")
        self.assertNotIn("gradient_clipping", health["alerts"])

    def test_nonfinite_checkpoint_is_rejected_before_live_policy_mutation(self):
        target = DrivingDQNAgent(_dqn(seed=31))
        source = DrivingDQNAgent(_dqn(seed=47))
        before = [
            parameter.detach().clone()
            for parameter in target.online_network.parameters()
        ]
        before_steps = target.environment_steps
        malformed = deepcopy(source.state_dict())
        first_key = next(iter(malformed["target_network"]))
        malformed["target_network"][first_key].reshape(-1)[0] = math.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            target.load_state_dict(malformed)

        self.assertEqual(target.environment_steps, before_steps)
        for expected, actual in zip(before, target.online_network.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_late_shape_failure_is_preflighted_before_online_weight_mutation(self):
        target = DrivingDQNAgent(_dqn(seed=31))
        source = DrivingDQNAgent(_dqn(seed=47))
        before = [
            parameter.detach().clone()
            for parameter in target.online_network.parameters()
        ]
        malformed = deepcopy(source.state_dict())
        key = next(iter(malformed["target_network"]))
        malformed["target_network"][key] = malformed["target_network"][key][:-1]

        with self.assertRaises(RuntimeError):
            target.load_state_dict(malformed)

        for expected, actual in zip(before, target.online_network.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_optimizer_scalars_and_incomplete_state_are_rejected_atomically(self):
        source = DrivingDQNAgent(_dqn(seed=47))
        state = np.zeros(16, dtype=np.float32)
        source.observe(state, 0, 1.0, state, False)
        source.observe(state, 0, 1.0, state, False)
        target = DrivingDQNAgent(_dqn(seed=31))
        before_lr = target.optimizer.param_groups[0]["lr"]

        nonfinite = deepcopy(source.state_dict())
        nonfinite["optimizer"]["param_groups"][0]["lr"] = math.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            target.load_state_dict(nonfinite)
        self.assertEqual(target.optimizer.param_groups[0]["lr"], before_lr)

        incomplete = deepcopy(source.state_dict())
        first_state = next(iter(incomplete["optimizer"]["state"].values()))
        first_state.pop("exp_avg_sq")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            target.load_state_dict(incomplete)
        self.assertEqual(target.optimizer.param_groups[0]["lr"], before_lr)


class PopulationHealthTests(unittest.TestCase):
    def _trainer(self, workers: int) -> PopulationTrainer:
        trainer = PopulationTrainer(
            _evolution(),
            _dqn(batch_size=1, warmup_steps=1),
            auto_evolve=False,
            parallel_workers=workers,
        )
        self.addCleanup(trainer.close)
        return trainer

    def test_health_counters_are_scored_and_deterministic_across_worker_counts(self):
        serial = self._trainer(1)
        parallel = self._trainer(2)

        serial.step_many(3)
        parallel.step_many(3)
        serial_health = serial.telemetry()["health"]
        parallel_health = parallel.telemetry()["health"]

        for key in ("replay", "optimization", "values", "safety"):
            self.assertEqual(serial_health[key], parallel_health[key])
        self.assertEqual(serial_health["optimization"]["decisions"], 6)
        self.assertEqual(serial_health["optimization"]["updates"], 6)
        self.assertEqual(serial_health["replay"]["size"], 6)
        self.assertEqual(serial_health["throughput"]["workers"], 1)
        self.assertEqual(parallel_health["throughput"]["workers"], 2)

    def test_worker_failure_is_a_critical_health_alert(self):
        trainer = self._trainer(1)
        trainer._worker_failure = ValueError("synthetic worker failure")

        health = trainer.telemetry()["health"]

        self.assertEqual(health["status"], "critical")
        self.assertTrue(health["throughput"]["worker_failed"])
        self.assertIn("worker_failure:ValueError", health["alerts"])

    def test_population_checkpoint_rejection_is_atomic(self):
        trainer = self._trainer(1)
        trainer.step_many(2)
        before_generation = trainer.generation
        before_ids = [member.member_id for member in trainer.population]
        before_weights = [
            [
                parameter.detach().clone()
                for parameter in member.agent.network.parameters()
            ]
            for member in trainer.population
        ]
        malformed = deepcopy(trainer.state_dict())
        agent_payload = malformed["population"][1]["agent"]
        target_key = next(iter(agent_payload["target_network"]))
        agent_payload["target_network"][target_key] = agent_payload["target_network"][
            target_key
        ][:-1]

        with self.assertRaises(RuntimeError):
            trainer.load_state_dict(malformed)

        self.assertEqual(trainer.generation, before_generation)
        self.assertEqual(
            [member.member_id for member in trainer.population], before_ids
        )
        for expected_member, actual_member in zip(before_weights, trainer.population):
            for expected, actual in zip(
                expected_member, actual_member.agent.network.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))

    def test_member_result_mismatch_is_rejected_before_population_commit(self):
        trainer = self._trainer(1)
        trainer.step_many(4)
        before_generation = trainer.generation
        before_ids = [member.member_id for member in trainer.population]
        malformed = deepcopy(trainer.state_dict())
        malformed["generation"] = before_generation + 9
        malformed["population"][0]["result"]["member_id"] = before_ids[1]

        with self.assertRaisesRegex(ValueError, "result member_id"):
            trainer.load_state_dict(malformed)

        self.assertEqual(trainer.generation, before_generation)
        self.assertEqual(
            [member.member_id for member in trainer.population], before_ids
        )

    def test_curriculum_and_counter_relations_are_strict(self):
        trainer = self._trainer(1)
        before_generation = trainer.generation

        malformed_flag = deepcopy(trainer.state_dict())
        malformed_flag["environment_curriculum"]["generation_ready"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            trainer.load_state_dict(malformed_flag)

        malformed_counters = deepcopy(trainer.state_dict())
        malformed_counters["health_counters"]["gradient_clip_events"] = 1
        with self.assertRaisesRegex(ValueError, "clips cannot exceed updates"):
            trainer.load_state_dict(malformed_counters)

        self.assertEqual(trainer.generation, before_generation)

    def test_population_telemetry_samples_environment_once(self):
        trainer = self._trainer(1)
        with patch.object(
            trainer.env,
            "telemetry",
            wraps=trainer.env.telemetry,
        ) as telemetry:
            trainer.telemetry()
        self.assertEqual(telemetry.call_count, 1)

    def test_session_surfaces_the_same_stable_health_contract(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="double_dqn",
                evaluation_steps=8,
                population_size=2,
                elite_count=1,
            ),
            dqn_config=_dqn(),
        )
        self.addCleanup(session.close)
        session.step_many(2)

        telemetry = session.telemetry()

        self.assertIn(
            telemetry["learning_status"],
            {"healthy", "warming_up", "warning", "critical"},
        )
        self.assertIs(telemetry["health"]["finite"], True)
        self.assertEqual(telemetry["health"]["optimization"]["decisions"], 2)

    def test_standalone_resume_window_keeps_clip_ratio_bounded(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(algorithm="double_dqn"),
            dqn_config=_dqn(),
        )
        self.addCleanup(session.close)
        session.agent.environment_steps = 10
        session.agent.gradient_steps = 10
        session.agent.gradient_clip_events = 5
        session._environment_decisions = 10
        session._health_decision_origin = 10
        session._health_update_origin = 10
        session._health_clip_origin = 5

        # First post-resume update is not clipped.
        session.agent.environment_steps += 1
        session.agent.gradient_steps += 1
        session._environment_decisions += 1
        health = session.telemetry()["health"]

        self.assertEqual(health["optimization"]["update_to_decision_ratio"], 1.0)
        self.assertEqual(health["optimization"]["clip_ratio"], 0.0)


class DrivingHealthInvariantTests(unittest.TestCase):
    def test_champion_snapshot_must_match_embedded_result(self):
        result = EvaluationResult(
            generation=2,
            member_id=7,
            fitness=3.5,
            total_reward=3.5,
            steps=4,
            laps=0,
            progress=0.2,
            collisions=0,
            terminated=False,
            truncated=True,
        )
        with self.assertRaisesRegex(ValueError, "generation"):
            ChampionSnapshot(1, 7, 3.5, result)
        with self.assertRaisesRegex(ValueError, "fitness"):
            ChampionSnapshot(2, 7, 4.5, result)

    def test_alert_code_is_rendered_as_an_actionable_reason(self):
        self.assertEqual(
            _health_alert_label("worker_failure:RuntimeError"),
            "WORKER FAIL",
        )
        self.assertEqual(
            _health_alert_label("high_wall_contact_rate"),
            "WALL CONTACT",
        )


if __name__ == "__main__":
    unittest.main()
