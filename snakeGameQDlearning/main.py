"""Train and inspect DQN or Double-DQN on the Snake environment."""

from __future__ import annotations

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Snake reinforcement-learning observatory")
    parser.add_argument("--algorithm", choices=("dqn", "double_dqn"), default="double_dqn")
    parser.add_argument("--games", type=int, default=0, help="Stop after N games; 0 runs continuously")
    parser.add_argument("--speed", type=int, default=120, help="Rendered frames per second")
    parser.add_argument("--headless", action="store_true", help="Train without showing a window")
    parser.add_argument("--fresh", action="store_true", help="Do not load the best checkpoint")
    parser.add_argument("--plot", action="store_true", help="Show the matplotlib training chart")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--screenshot", help="Save an inspector PNG and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.headless or args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame
    from snakeGameQDlearning.src.game import SnakeGameAI
    from snakeGameQDlearning.src.ml.agent import Agent

    agent = Agent(algorithm=args.algorithm, seed=args.seed)
    game = SnakeGameAI(render=not (args.headless or args.screenshot), speed=args.speed, seed=args.seed)
    if not args.fresh and not args.screenshot:
        agent.load_best_model()
    starting_game_count = agent.n_games

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
                game.reset()
                agent.n_games += 1
                agent.previous_distances.clear()
                episode_return = 0.0
        state = agent.get_state(game)
        agent.get_action(state)
        game.set_debug_info(**agent.telemetry(state, game, episode_return))
        game.save_screenshot(args.screenshot)
        pygame.quit()
        return

    print(f"Training {args.algorithm.upper()} — Esc or Ctrl+C to stop")
    scores, mean_scores = [], []
    total_score = 0
    record = agent.get_loaded_best_score()
    best_mean = agent.get_loaded_mean_score()
    episode_return = 0.0

    try:
        while game.running and (args.games == 0 or agent.n_games - starting_game_count < args.games):
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
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                agent.previous_distances.clear()
                scores.append(score)
                total_score += score
                mean_score = total_score / len(scores)
                mean_scores.append(mean_score)
                if score > record:
                    record = score
                    best_mean = max(best_mean, mean_score)
                    agent.save_model_new_record(record, mean_score)
                elif best_mean > 0 and mean_score > best_mean * 1.05:
                    best_mean = mean_score
                    agent.update_model_mean_score(mean_score)
                print(f"game={agent.n_games:4d} score={score:3d} record={record:3d} "
                      f"mean={mean_score:6.2f} memory={len(agent.memory):6d} "
                      f"return={episode_return:+7.2f} loss={agent.trainer.last_loss:.5f}")
                episode_return = 0.0
                if args.plot:
                    from snakeGameQDlearning.src.utils import plot_training_progress
                    plot_training_progress(scores, mean_scores)
    except KeyboardInterrupt:
        pass
    finally:
        session_games = agent.n_games - starting_game_count
        print(f"Stopped after {session_games} session games ({agent.n_games} total); "
              f"record={record}; memory={len(agent.memory)}")
        pygame.quit()


if __name__ == "__main__":
    main()
