"""Regression contract for lap-learning and population end diagnostics."""

from __future__ import annotations

import unittest

from drivingGameRL.src.ml import DQNConfig
from drivingGameRL.src.ml.evolution import (
    EvaluationResult,
    EvolutionConfig,
    GenerationRecord,
    PopulationTrainer,
)


def _dqn() -> DQNConfig:
    return DQNConfig(
        hidden_sizes=(8,),
        replay_capacity=16,
        batch_size=1,
        warmup_steps=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        seed=83,
    )


def _evolution() -> EvolutionConfig:
    return EvolutionConfig(
        algorithm="genetic",
        population_size=4,
        elite_count=1,
        tournament_size=2,
        evaluation_steps=8,
        mutation_rate=0.0,
        mutation_std=0.0,
        seed=83,
    )


class DrivingGenerationDiagnosticsTests(unittest.TestCase):
    def test_progress_and_end_reason_diagnostics_reject_malformed_values(self):
        with self.assertRaisesRegex(ValueError, "smaller than progress"):
            EvaluationResult(
                generation=0,
                member_id=0,
                fitness=1.0,
                total_reward=1.0,
                steps=1,
                laps=0,
                progress=0.8,
                collisions=0,
                terminated=False,
                truncated=True,
                max_progress=0.7,
            )

        record = {
            "generation": 0,
            "best_fitness": 1.0,
            "mean_fitness": 0.5,
            "median_fitness": 0.5,
            "worst_fitness": 0.0,
            "fitness_std": 0.5,
            "champion_id": 0,
            "elite_ids": [0],
            "population_size": 2,
            "genome_diversity": 0.1,
            "end_reasons": (("stagnation", 1), ("stagnation", 1)),
        }
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            GenerationRecord.from_dict(record)

    def test_generation_exposes_laps_progress_near_finishes_and_end_reasons(self):
        trainer = PopulationTrainer(
            _evolution(),
            _dqn(),
            auto_evolve=False,
            parallel_workers=1,
        )
        self.addCleanup(trainer.close)

        cases = (
            # A recovered finisher.
            (410.0, 1, 1.0, 1.0, 1, True, False, "lap_completed"),
            # A useful near-finish that should be visible instead of disappearing
            # into one scalar fitness number.
            (275.0, 0, 0.70, 0.96, 2, False, True, "collision_loop"),
            (115.0, 0, 0.62, 0.62, 0, False, True, "stagnation"),
            (80.0, 0, 0.18, 0.18, 0, False, True, "step_limit"),
        )
        for member, runtime, values in zip(
            trainer.population,
            trainer._member_runtimes,
            cases,
        ):
            (
                fitness,
                laps,
                progress,
                max_progress,
                collisions,
                terminated,
                truncated,
                reason,
            ) = values
            member.result = EvaluationResult(
                generation=trainer.generation,
                member_id=member.member_id,
                fitness=fitness,
                total_reward=fitness,
                steps=8,
                laps=laps,
                progress=progress,
                collisions=collisions,
                terminated=terminated,
                truncated=truncated,
                end_reason=reason,
                collision_recoveries=1 if laps > 0 else 0,
                max_progress=max_progress,
            )
            runtime.last_info = {
                "lap_completed": laps > 0,
                "truncation_reason": None if laps > 0 else reason,
                "collision_recoveries": 1 if laps > 0 else 0,
            }

        telemetry = trainer.telemetry()
        diagnostics = telemetry["generation_metrics"]

        self.assertEqual(diagnostics["evaluated_members"], 4)
        self.assertEqual(diagnostics["laps_completed"], 1)
        self.assertAlmostEqual(diagnostics["lap_completion_rate"], 0.25)
        self.assertEqual(diagnostics["best_progress"], 1.0)
        self.assertAlmostEqual(diagnostics["mean_progress"], 0.69)
        self.assertEqual(diagnostics["near_finish_threshold"], 0.90)
        self.assertEqual(diagnostics["near_finish_count"], 1)
        self.assertEqual(diagnostics["collision_recoveries"], 1)
        self.assertEqual(
            diagnostics["end_reasons"],
            {
                "lap_completed": 1,
                "collision_loop": 1,
                "stagnation": 1,
                "step_limit": 1,
            },
        )

        partial_checkpoint = trainer.state_dict()
        partial_restored = PopulationTrainer(
            _evolution(),
            _dqn(),
            auto_evolve=False,
            parallel_workers=1,
        )
        self.addCleanup(partial_restored.close)
        partial_restored.load_state_dict(partial_checkpoint)
        self.assertEqual(
            partial_restored.telemetry()["generation_metrics"],
            diagnostics,
        )

        # The same learning evidence must survive the immediate generation
        # reset, history rendering, and checkpoint round trips via the record.
        record = trainer.evolve()
        record_values = record.to_dict()
        self.assertEqual(record_values["laps_completed"], 1)
        self.assertAlmostEqual(record_values["lap_completion_rate"], 0.25)
        self.assertEqual(record_values["best_progress"], 1.0)
        self.assertAlmostEqual(record_values["mean_progress"], 0.69)
        self.assertEqual(record_values["near_finish_count"], 1)
        self.assertEqual(record_values["collision_recoveries"], 1)
        self.assertEqual(record_values["end_reasons"], diagnostics["end_reasons"])
        self.assertEqual(trainer.telemetry()["history"][-1], record_values)

        checkpoint = trainer.state_dict()
        restored = PopulationTrainer(
            _evolution(),
            _dqn(),
            auto_evolve=False,
            parallel_workers=1,
        )
        self.addCleanup(restored.close)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.telemetry()["history"][-1], record_values)


if __name__ == "__main__":
    unittest.main()
