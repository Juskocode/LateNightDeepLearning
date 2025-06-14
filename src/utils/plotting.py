import matplotlib.pyplot as plt
from matplotlib import style
from typing import List
import os

# Set matplotlib style
style.use('ggplot')


def plot_training_progress(scores: List[int], mean_scores: List[float]) -> None:
    plt.figure(figsize=(12, 6))

    # Clear the current plot
    plt.clf()

    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot scores and mean scores
    plt.plot(scores, label='Score', alpha=0.7, color='blue')
    plt.plot(mean_scores, label='Mean Score', color='red', linewidth=2)

    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Show current values in the title
    if scores:
        plt.title(f'Training Progress - Current Score: {scores[-1]}, '
                  f'Mean Score: {mean_scores[-1]:.2f}')

    # Update plot
    plt.pause(0.1)
    plt.show(block=False)


def save_training_plot(scores: List[int], mean_scores: List[float],
                       filename: str = 'training_progress.png') -> None:
    plt.figure(figsize=(12, 6))

    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label='Score', alpha=0.7, color='blue')
    plt.plot(mean_scores, label='Mean Score', color='red', linewidth=2)

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()