from src.ml.agent import Agent
from src.game.snake_game import SnakeGameAI
from src.utils.plotting import plot_training_progress


def main():
    print("Starting Snake AI Training...")

    # Initialize components
    agent = Agent()
    game = SnakeGameAI()

    # Try to load existing model
    try:
        agent.load_model()
        print("Successfully loaded existing model. Continuing training...")
    except FileNotFoundError:
        print("No existing model found. Starting fresh training...")
    except Exception as e:
        print(f"Error loading model: {e}. Starting fresh training...")

    # Training metrics
    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    try:
        while True:
            # Get current state
            state_old = agent.get_state(game)

            # Get action from agent
            action = agent.get_action(state_old)

            # Perform action and get new state
            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)

            # Train on this step
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            if done:
                # Game over - train on batch and reset
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # Update metrics
                if score > record:
                    record = score
                    agent.save_model()

                print(f'Game {agent.n_games}, Score: {score}, Record: {record}')

                # Update plotting data
                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                mean_scores.append(mean_score)

                # Plot progress
                plot_training_progress(scores, mean_scores)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training stopped due to error: {e}")
    finally:
        print("Training session ended")


if __name__ == '__main__':
    main()