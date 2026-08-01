"""Train, compare, and inspect deep or tabular RL on Snake."""

from __future__ import annotations

import argparse
import json
import os


DEFAULT_VALIDATION_EPISODES = 8
DEFAULT_FINAL_TEST_EPISODES = 20


def parse_args():
    from snakeGameQDlearning.src.game.environments import available_environments
    from snakeGameQDlearning.src.ml.algorithms import available_algorithms

    parser = argparse.ArgumentParser(
        description="Snake reinforcement-learning observatory"
    )
    parser.add_argument(
        "--algorithm", choices=available_algorithms(), default="double_dqn"
    )
    parser.add_argument(
        "--environment",
        choices=available_environments(),
        default="standard",
        help="Educational board preset",
    )
    parser.add_argument(
        "--games", type=int, default=0, help="Stop after N games; 0 runs continuously"
    )
    parser.add_argument(
        "--speed", type=int, default=120, help="Rendered frames per second"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Train without showing a window"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Do not load the best checkpoint"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not write checkpoints"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show the matplotlib training chart"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        help="Root for the fixed validation suite; defaults to a separate stream derived from --seed",
    )
    parser.add_argument(
        "--final-test-seed",
        type=int,
        help="Root for the fixed final-test suite; defaults to a third stream derived from --seed",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=25,
        help="Run fixed-suite validation every N completed games; 0 disables it",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        help=f"Validation episodes (default: {DEFAULT_VALIDATION_EPISODES}; stored count for checkpoint reproduction)",
    )
    parser.add_argument(
        "--final-test-episodes",
        type=int,
        help=f"Final-test episodes (default: {DEFAULT_FINAL_TEST_EPISODES})",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a checkpoint, run the selected greedy evaluation suite, and exit",
    )
    parser.add_argument(
        "--eval-suite",
        choices=("validation", "final_test"),
        default="validation",
        help="Suite used by --eval-only; final_test is never used for checkpoint selection",
    )
    parser.add_argument(
        "--domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Vary valid starting positions and headings between training episodes",
    )
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Increase spawn randomization gradually during training",
    )
    parser.add_argument("--screenshot", help="Save an inspector PNG and exit")
    args = parser.parse_args()
    if args.eval_every < 0:
        parser.error("--eval-every must be non-negative")
    if args.eval_episodes is not None and args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.final_test_episodes is not None and args.final_test_episodes <= 0:
        parser.error("--final-test-episodes must be positive")
    return args


def main():
    args = parse_args()
    if args.headless or args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame
    from snakeGameQDlearning.src.game import (
        EpisodeSeedStreams,
        SnakeCurriculum,
        SnakeGameAI,
        get_environment_preset,
    )
    from snakeGameQDlearning.src.ml.agent import Agent
    from snakeGameQDlearning.src.ml.evaluation import (
        evaluate_agent,
        resolve_evaluation_suite,
    )

    agent = Agent(algorithm=args.algorithm, seed=args.seed)
    preset = get_environment_preset(args.environment)
    validation_override = (
        args.evaluation_seed is not None or args.eval_episodes is not None
    )
    final_test_override = (
        args.final_test_seed is not None or args.final_test_episodes is not None
    )
    validation_episode_count = args.eval_episodes or DEFAULT_VALIDATION_EPISODES
    final_test_episode_count = args.final_test_episodes or DEFAULT_FINAL_TEST_EPISODES
    seed_streams = EpisodeSeedStreams(
        args.seed, args.evaluation_seed, args.final_test_seed
    )
    validation_seeds = seed_streams.validation_seeds(validation_episode_count)
    final_test_seeds = seed_streams.final_test_seeds(final_test_episode_count)
    curriculum = SnakeCurriculum()

    def episode_setup(episode: int) -> tuple[int, bool, str]:
        episode_seed = seed_streams.train_seed(episode)
        stage = curriculum.stage_for(episode)
        if not args.domain_randomization:
            randomized = False
            stage_name = "fixed_spawn"
        elif args.curriculum:
            randomized = curriculum.randomize_start(episode, episode_seed)
            stage_name = stage.name
        else:
            randomized = True
            stage_name = "full_randomization"
        return episode_seed, randomized, stage_name

    initial_seed, initial_randomized, initial_stage = episode_setup(agent.n_games)
    agent.curriculum_stage = initial_stage
    game = SnakeGameAI(
        width=preset.width,
        height=preset.height,
        render=not (args.headless or args.screenshot or args.eval_only),
        speed=args.speed,
        seed=initial_seed,
        randomize_start=initial_randomized,
    )
    loaded_checkpoint = False
    if not args.fresh and not args.screenshot:
        loaded_checkpoint = agent.load_best_model(
            environment=preset.name,
            validation_seeds=validation_seeds if validation_override else None,
        )
        if args.eval_only and validation_override and not loaded_checkpoint:
            # Explicit evaluation overrides describe where to test a compatible
            # policy, not which historical validation suite trained it.
            loaded_checkpoint = agent.load_best_model(environment=preset.name)
    if args.eval_only and not args.fresh and not loaded_checkpoint:
        pygame.quit()
        raise SystemExit(
            f"No compatible {agent.algorithm} checkpoint was found. "
            "Use --fresh to evaluate an intentionally untrained policy."
        )
    if loaded_checkpoint and agent.loaded_metadata:
        stored_experiment = agent.loaded_metadata.get("experiment", {})
        compatible_experiment = (
            isinstance(stored_experiment, dict)
            and stored_experiment.get("environment") == preset.name
        )
        if compatible_experiment and not validation_override:
            stored_validation = stored_experiment.get("validation_seeds")
            if isinstance(stored_validation, list) and stored_validation:
                validation_seeds = tuple(int(seed) for seed in stored_validation)
                validation_episode_count = len(validation_seeds)
                stored_root = stored_experiment.get("validation_seed_root")
                if isinstance(stored_root, int):
                    seed_streams.evaluation_seed = stored_root
        if compatible_experiment and not final_test_override:
            stored_final_test = stored_experiment.get("final_test_seeds")
            if isinstance(stored_final_test, list) and stored_final_test:
                final_test_seeds = tuple(int(seed) for seed in stored_final_test)
                final_test_episode_count = len(final_test_seeds)
                stored_root = stored_experiment.get("final_test_seed_root")
                if isinstance(stored_root, int):
                    seed_streams.final_test_seed = stored_root
    starting_game_count = agent.n_games
    loaded_seed, loaded_randomized, loaded_stage = episode_setup(agent.n_games)
    agent.curriculum_stage = loaded_stage
    game.reset(seed=loaded_seed, randomize_start=loaded_randomized)

    def run_evaluation(seeds: tuple[int, ...], training_mean: float) -> dict:
        result = evaluate_agent(agent, seeds, preset)
        metrics = result.as_dict()
        agent.update_evaluation_metrics(metrics, training_mean)
        return agent.evaluation_metrics

    def experiment_metadata(evaluation_round: int | None) -> dict:
        """Configuration needed to compare or reproduce saved evaluations."""

        return {
            "schema_version": 1,
            "environment": preset.name,
            "environment_size": [preset.width, preset.height],
            "training_seed_root": args.seed,
            "validation_seed_root": seed_streams.evaluation_seed,
            "validation_seeds": list(validation_seeds),
            "validation_episodes": validation_episode_count,
            "final_test_seed_root": seed_streams.final_test_seed,
            "final_test_seeds": list(final_test_seeds),
            "final_test_episodes": final_test_episode_count,
            "evaluation_round": evaluation_round,
            "evaluation_every": args.eval_every,
            "domain_randomization": args.domain_randomization,
            "curriculum": args.curriculum,
        }

    if args.eval_only:
        requested_count = (
            validation_episode_count
            if args.eval_suite == "validation"
            else final_test_episode_count
        )
        explicit_override = (
            validation_override
            if args.eval_suite == "validation"
            else final_test_override
        )
        selection = resolve_evaluation_suite(
            seed_streams,
            args.eval_suite,
            requested_count,
            loaded_metadata=agent.loaded_metadata,
            environment=preset.name,
            prefer_stored=loaded_checkpoint and not explicit_override,
        )
        metrics = run_evaluation(selection.seeds, agent.get_loaded_mean_score())
        payload = {
            "algorithm": agent.algorithm,
            "environment": preset.name,
            "suite": selection.name,
            "suite_source": selection.source,
            "suite_seed_root": selection.seed_root,
            "suite_seeds": list(selection.seeds),
            "checkpoint_evaluation_round": selection.evaluation_round,
            **metrics,
        }
        if (
            selection.name == "validation"
            and selection.source == "checkpoint"
            and agent.get_loaded_evaluation_mean() is not None
        ):
            stored_mean = agent.get_loaded_evaluation_mean()
            payload["checkpoint_validation_mean"] = stored_mean
            payload["reproduces_checkpoint_mean"] = (
                abs(float(metrics["mean_score"]) - float(stored_mean)) < 1e-12
            )
        print(json.dumps(payload, sort_keys=True))
        pygame.quit()
        return

    state = agent.get_state(game)
    action = agent.get_action(state)
    game.set_debug_info(**agent.telemetry(state, game))
    if args.screenshot:
        # Populate the observatory with real transitions, gradients, target
        # values, and replay history instead of documenting an empty buffer.
        episode_return = 0.0
        for _ in range(32):
            state_old = agent.get_state(game)
            old_score = game.score
            action = agent.get_action(state_old)
            _, done, score = game.play_step(action, render_frame=False)
            state_new = agent.get_state(game)
            reward = agent.calculate_reward(game, done, score, old_score)
            episode_return += reward
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            if done:
                agent.n_games += 1
                agent.previous_distances.clear()
                episode_return = 0.0
                next_seed, randomized, stage_name = episode_setup(agent.n_games)
                agent.curriculum_stage = stage_name
                game.reset(seed=next_seed, randomize_start=randomized)
        # The documentation capture should demonstrate the same held-out
        # generalization measurement as a live training session, not a
        # placeholder panel populated only with training transitions.
        run_evaluation(validation_seeds, float(game.score))
        state = agent.get_state(game)
        agent.get_action(state)
        game.set_debug_info(**agent.telemetry(state, game, episode_return))
        game.save_screenshot(args.screenshot)
        pygame.quit()
        return

    print(
        f"Training {agent.algorithm.upper()} on {preset.name} — Esc or Ctrl+C to stop\n"
        f"train seed root={args.seed}; validation root={seed_streams.evaluation_seed}; "
        f"final-test root={seed_streams.final_test_seed}; "
        f"domain randomization={'on' if args.domain_randomization else 'off'}"
    )
    scores, mean_scores = [], []
    total_score = 0
    record = agent.get_loaded_best_score()
    best_mean = agent.get_loaded_mean_score()
    loaded_evaluation_mean = agent.get_loaded_evaluation_mean()
    loaded_experiment = (
        dict(agent.loaded_metadata.get("experiment", {}))
        if agent.loaded_metadata
        and isinstance(agent.loaded_metadata.get("experiment", {}), dict)
        else {}
    )
    same_validation_suite = loaded_experiment.get(
        "environment"
    ) == preset.name and loaded_experiment.get("validation_seeds") == list(
        validation_seeds
    )
    best_evaluation = (
        loaded_evaluation_mean
        if loaded_evaluation_mean is not None and same_validation_suite
        else float("-inf")
    )
    stored_round = loaded_experiment.get("evaluation_round")
    latest_evaluation_round = (
        int(stored_round)
        if same_validation_suite and isinstance(stored_round, int)
        else None
    )
    episode_return = 0.0

    try:
        while game.running and (
            args.games == 0 or agent.n_games - starting_game_count < args.games
        ):
            if game.paused:
                game.play_step(None, render_frame=not args.headless)
                continue
            state_old = agent.get_state(game)
            old_score = game.score
            action = agent.get_action(state_old)
            game.set_debug_info(**agent.telemetry(state_old, game, episode_return))
            if not args.headless:
                game.render()
            _, done, score = game.play_step(action, render_frame=False)
            if not game.transition_applied:
                continue
            state_new = agent.get_state(game)
            reward = agent.calculate_reward(game, done, score, old_score)
            episode_return += reward
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            game.set_debug_info(**agent.telemetry(state_old, game, episode_return))

            if done and game.running:
                if not args.headless:
                    game.render()
                agent.n_games += 1
                agent.train_long_memory()
                agent.previous_distances.clear()
                scores.append(score)
                total_score += score
                mean_score = total_score / len(scores)
                mean_scores.append(mean_score)
                rolling_train_mean = sum(scores[-100:]) / min(100, len(scores))
                evaluation_improved = False
                evaluation_this_episode = None
                if args.eval_every > 0 and agent.n_games % args.eval_every == 0:
                    evaluation_round = agent.n_games // args.eval_every
                    latest_evaluation_round = evaluation_round
                    evaluation = run_evaluation(validation_seeds, rolling_train_mean)
                    print(
                        f"validation round={evaluation_round} "
                        f"episodes={evaluation['episodes']} "
                        f"mean={evaluation['mean_score']:.2f} "
                        f"std={evaluation['std_score']:.2f} "
                        f"gap={evaluation['generalization_gap']:+.2f}"
                    )
                    if evaluation["mean_score"] > best_evaluation:
                        best_evaluation = evaluation["mean_score"]
                        evaluation_improved = True
                    evaluation_this_episode = evaluation["mean_score"]
                new_record = score > record
                mean_improved = False
                if new_record:
                    record = score
                    best_mean = max(best_mean, mean_score)
                elif best_mean > 0 and mean_score > best_mean * 1.05:
                    best_mean = mean_score
                    mean_improved = True
                if not args.no_save and (
                    new_record or mean_improved or evaluation_improved
                ):
                    reasons = []
                    if new_record:
                        reasons.append("training_record")
                    if mean_improved:
                        reasons.append("training_mean")
                    if evaluation_improved:
                        reasons.append("held_out_validation")
                    agent.save_model_checkpoint(
                        record,
                        mean_score,
                        reason="_and_".join(reasons),
                        evaluation_mean=evaluation_this_episode,
                        experiment=experiment_metadata(latest_evaluation_round),
                    )
                print(
                    f"game={agent.n_games:4d} score={score:3d} record={record:3d} "
                    f"mean={mean_score:6.2f} memory={len(agent.memory):6d} "
                    f"return={episode_return:+7.2f} loss={agent.learning.last_loss:.5f}"
                )
                episode_return = 0.0
                next_seed, randomized, stage_name = episode_setup(agent.n_games)
                agent.curriculum_stage = stage_name
                game.reset(seed=next_seed, randomize_start=randomized)
                if args.plot:
                    from snakeGameQDlearning.src.utils import plot_training_progress

                    plot_training_progress(scores, mean_scores)
    except KeyboardInterrupt:
        pass
    finally:
        session_games = agent.n_games - starting_game_count
        print(
            f"Stopped after {session_games} session games ({agent.n_games} total); "
            f"record={record}; memory={len(agent.memory)}"
        )
        pygame.quit()


if __name__ == "__main__":
    main()
