from snakeGameQDlearning.src.ml.agent import Agent
from snakeGameQDlearning.src.game import SnakeGameAI
from snakeGameQDlearning.src.utils import plot_training_progress


def main():
    print("Starting Snake AI Training...")

    agent = Agent()
    game = SnakeGameAI()

    model_loaded = False
    try:
        model_loaded = agent.load_best_model()
        if not model_loaded:
            print("No existing models found. Starting fresh training...")
    except Exception as e:
        print(f"Error loading model: {e}. Starting fresh training...")

    scores = []
    mean_scores = []
    total_score = 0
    record = agent.get_loaded_best_score() if model_loaded else 0
    best_mean_score = agent.get_loaded_mean_score() if model_loaded else 0
    print(f"Starting with record: {record}, best mean: {best_mean_score:.2f}")

    try:
        while True:
            state_old = agent.get_state(game)
            old_score = game.score
            action = agent.get_action(state_old)
            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)

            improved_reward = agent.calculate_reward(game, done, score, old_score)

            agent.train_short_memory(state_old, action, improved_reward, state_new, done)
            agent.remember(state_old, action, improved_reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                agent.previous_distances.clear()

                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                mean_scores.append(mean_score)

                # Check for new high score (save new version)
                if score > record:
                    record = score
                    best_mean_score = mean_score
                    agent.save_model_new_record(record, mean_score)
                # Check for 5% improvement in mean score (update existing model)
                elif mean_score > best_mean_score * 1.05:
                    best_mean_score = mean_score
                    agent.update_model_mean_score(mean_score)
                    print(f"Mean score improved by 5%! New mean: {mean_score:.2f}")

                print(f'Game {agent.n_games}, Score: {score}, Record: {record}, Mean: {mean_score:.2f}')

                plot_training_progress(scores, mean_scores)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Final stats - Games: {agent.n_games}, Record: {record}, Best Mean: {best_mean_score:.2f}")
    except Exception as e:
        print(f"Training stopped due to error: {e}")
    finally:
        print("Training session ended")


if __name__ == '__main__':
    main()