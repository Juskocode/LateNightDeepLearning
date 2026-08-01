import tempfile
import unittest
from pathlib import Path

import torch

from drivingGameRL.src.ml import DQNConfig
from drivingGameRL.src.ml.evolution import (
    EvaluationResult,
    EvolutionConfig,
    PopulationTrainer,
)


def _dqn_config(seed=11):
    return DQNConfig(
        hidden_sizes=(8,),
        replay_capacity=16,
        batch_size=1,
        warmup_steps=1,
        target_sync_interval=2,
        epsilon_start=0.0,
        epsilon_end=0.0,
        seed=seed,
    )


def _evolution_config(**changes):
    values = {
        "algorithm": "genetic",
        "population_size": 4,
        "elite_count": 1,
        "tournament_size": 2,
        "evaluation_steps": 1,
        "mutation_rate": 0.0,
        "mutation_std": 0.0,
        "history_capacity": 4,
        "seed": 23,
    }
    values.update(changes)
    return EvolutionConfig(**values)


def _parameters(agent):
    return torch.cat(
        [
            parameter.detach().reshape(-1).cpu()
            for parameter in agent.online_network.parameters()
        ]
    )


def _set_result(trainer, index, fitness):
    member = trainer.population[index]
    member.result = EvaluationResult(
        generation=trainer.generation,
        member_id=member.member_id,
        fitness=float(fitness),
        total_reward=float(fitness),
        steps=1,
        laps=0,
        progress=0.01 * index,
        collisions=0,
        terminated=False,
        truncated=True,
    )


class DrivingEvolutionTests(unittest.TestCase):
    def test_seeded_populations_are_reproducible_but_members_are_distinct(self):
        config = _evolution_config()
        first = PopulationTrainer(config, _dqn_config(), auto_evolve=False)
        second = PopulationTrainer(config, _dqn_config(), auto_evolve=False)

        for first_member, second_member in zip(first.population, second.population):
            self.assertTrue(
                torch.equal(
                    _parameters(first_member.agent),
                    _parameters(second_member.agent),
                )
            )
        self.assertFalse(
            torch.equal(
                _parameters(first.population[0].agent),
                _parameters(first.population[1].agent),
            )
        )

    def test_tournament_selection_uses_fitness_and_stable_id_tie_break(self):
        trainer = PopulationTrainer(
            _evolution_config(tournament_size=4),
            _dqn_config(),
            auto_evolve=False,
        )
        for index, fitness in enumerate((1.0, 8.0, 8.0, -2.0)):
            _set_result(trainer, index, fitness)

        selected = trainer.tournament_select()
        self.assertEqual(selected.member_id, trainer.population[1].member_id)

    def test_uniform_and_blend_crossover_use_both_parent_genomes(self):
        uniform = PopulationTrainer(
            _evolution_config(
                population_size=2,
                tournament_size=2,
                crossover="uniform",
                crossover_rate=1.0,
            ),
            _dqn_config(),
            auto_evolve=False,
        )
        first, second = (member.agent for member in uniform.population)
        with torch.no_grad():
            for parameter in first.online_network.parameters():
                parameter.zero_()
            for parameter in second.online_network.parameters():
                parameter.fill_(1.0)
        child = uniform.crossover_agents(first, second, seed=90)
        genes = _parameters(child)
        self.assertTrue(torch.all((genes == 0.0) | (genes == 1.0)))
        self.assertTrue(torch.any(genes == 0.0))
        self.assertTrue(torch.any(genes == 1.0))

        blend = PopulationTrainer(
            _evolution_config(
                population_size=2,
                tournament_size=2,
                crossover="blend",
                crossover_rate=1.0,
                blend_alpha=0.25,
            ),
            _dqn_config(),
            auto_evolve=False,
        )
        blend_first, blend_second = (member.agent for member in blend.population)
        with torch.no_grad():
            for parameter in blend_first.online_network.parameters():
                parameter.zero_()
            for parameter in blend_second.online_network.parameters():
                parameter.fill_(1.0)
        blend_child = blend.crossover_agents(blend_first, blend_second, seed=91)
        blend_genes = _parameters(blend_child)
        self.assertGreaterEqual(float(blend_genes.min()), -0.25)
        self.assertLessEqual(float(blend_genes.max()), 1.25)
        self.assertTrue(torch.any((blend_genes != 0.0) & (blend_genes != 1.0)))

    def test_masked_gaussian_mutation_is_deterministic_and_syncs_target(self):
        config = _evolution_config(
            population_size=2,
            tournament_size=2,
            mutation_rate=1.0,
            mutation_std=0.1,
        )
        first = PopulationTrainer(config, _dqn_config(), auto_evolve=False)
        second = PopulationTrainer(config, _dqn_config(), auto_evolve=False)
        first_agent = first.population[0].agent
        second_agent = second.population[0].agent

        first_count = first.mutate_agent(first_agent)
        second_count = second.mutate_agent(second_agent)
        self.assertEqual(first_count, first_agent.online_network.parameter_count)
        self.assertEqual(first_count, second_count)
        self.assertTrue(
            torch.equal(_parameters(first_agent), _parameters(second_agent))
        )
        self.assertTrue(
            all(
                torch.equal(online, target)
                for online, target in zip(
                    first_agent.online_network.parameters(),
                    first_agent.target_network.parameters(),
                )
            )
        )

    def test_generation_advance_preserves_strict_elites_and_lineage(self):
        trainer = PopulationTrainer(
            _evolution_config(elite_count=2),
            _dqn_config(),
            auto_evolve=False,
        )
        for index, fitness in enumerate((1.0, 7.0, 2.0, 9.0)):
            _set_result(trainer, index, fitness)
        expected_ids = (
            trainer.population[3].member_id,
            trainer.population[1].member_id,
        )
        elite_weights = {
            member_id: _parameters(trainer.population[member_id].agent).clone()
            for member_id in expected_ids
        }

        record = trainer.evolve()

        self.assertEqual(record.elite_ids, expected_ids)
        self.assertEqual(trainer.generation, 1)
        self.assertEqual(len(trainer.history), 1)
        self.assertEqual(
            tuple(member.member_id for member in trainer.population[:2]), expected_ids
        )
        for elite in trainer.population[:2]:
            self.assertTrue(
                torch.equal(_parameters(elite.agent), elite_weights[elite.member_id])
            )
            self.assertIsNone(elite.result)
            self.assertEqual(elite.parent_ids, (elite.member_id,))
        self.assertTrue(
            all(member.birth_generation == 1 for member in trainer.population[2:])
        )

    def test_champion_race_clone_is_isolated_from_population(self):
        trainer = PopulationTrainer(
            _evolution_config(), _dqn_config(), auto_evolve=False
        )
        for index, fitness in enumerate((1.0, 3.0, 9.0, 2.0)):
            _set_result(trainer, index, fitness)
        trainer.evolve()
        expected_id = 2
        self.assertEqual(trainer.champion_snapshot.member_id, expected_id)

        race_agent = trainer.champion_agent()
        original = _parameters(trainer.champion_agent()).clone()
        with torch.no_grad():
            for parameter in race_agent.online_network.parameters():
                parameter.add_(100.0)
        self.assertTrue(torch.equal(_parameters(trainer.champion_agent()), original))
        self.assertFalse(torch.equal(_parameters(race_agent), original))

    def test_hybrid_evaluation_performs_td_learning_while_pure_genetic_does_not(self):
        pure = PopulationTrainer(
            _evolution_config(
                algorithm="genetic", population_size=2, tournament_size=2
            ),
            _dqn_config(),
            auto_evolve=False,
        )
        hybrid = PopulationTrainer(
            _evolution_config(
                algorithm="genetic_dqn", population_size=2, tournament_size=2
            ),
            _dqn_config(),
            auto_evolve=False,
        )
        pure_before = _parameters(pure.population[0].agent).clone()
        hybrid_before = _parameters(hybrid.population[0].agent).clone()
        self.assertTrue(torch.equal(pure_before, hybrid_before))

        pure.step()
        hybrid.step()

        self.assertTrue(torch.equal(_parameters(pure.population[0].agent), pure_before))
        self.assertFalse(
            torch.equal(_parameters(hybrid.population[0].agent), hybrid_before)
        )
        self.assertEqual(len(pure.population[0].agent.replay), 0)
        self.assertEqual(len(hybrid.population[0].agent.replay), 1)
        self.assertEqual(pure.population[0].agent.gradient_steps, 0)
        self.assertEqual(hybrid.population[0].agent.gradient_steps, 1)

    def test_history_is_bounded_and_checkpoint_keeps_champion_genome(self):
        config = _evolution_config(history_capacity=2)
        trainer = PopulationTrainer(config, _dqn_config(), auto_evolve=False)
        for generation in range(3):
            for index in range(len(trainer.population)):
                _set_result(trainer, index, generation * 10 + index)
            trainer.evolve()
        self.assertEqual([record.generation for record in trainer.history], [1, 2])
        champion_weights = _parameters(trainer.champion_agent(best_ever=True)).clone()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "population.pth")
            restored = PopulationTrainer(config, _dqn_config(), auto_evolve=False)
            restored.load(checkpoint)

        self.assertEqual(restored.generation, trainer.generation)
        self.assertEqual(
            restored.best_champion_snapshot, trainer.best_champion_snapshot
        )
        self.assertTrue(
            torch.equal(
                _parameters(restored.champion_agent(best_ever=True)),
                champion_weights,
            )
        )
        self.assertLessEqual(
            sum(len(member.agent.replay) for member in restored.population),
            config.population_size * restored.dqn_config.replay_capacity,
        )


if __name__ == "__main__":
    unittest.main()
