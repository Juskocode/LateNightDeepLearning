from src.ml.agent import Agent
from src.game.snake_game import SnakeGameAI
from src.utils.plotting import plot_training_progress


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
    print(f"Starting with record: {record}")

    try:
        while True:
            state_old = agent.get_state(game)
            action = agent.get_action(state_old)
            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.save_model(score)
                    print(f"New record! Saved model with score: {score}")

                print(f'Game {agent.n_games}, Score: {score}, Record: {record}')

                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                mean_scores.append(mean_score)

                plot_training_progress(scores, mean_scores)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Final stats - Games: {agent.n_games}, Record: {record}")
    except Exception as e:
        print(f"Training stopped due to error: {e}")
    finally:
        print("Training session ended")


if __name__ == '__main__':
    main()