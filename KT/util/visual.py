import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

def plot_heatmap(matrix, title='Heatmap Example'):
    # Plot the heatmap
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a colorbar to show the scale
    plt.title(title)
    plt.show()

def plot_multiple_heatmap(matrix_list, title='Multiple Heatmap Example'):
    # Create subplots with 1 row and 3 columns
    n_matrix = len(matrix_list)
    fig, axes = plt.subplots(1,n_matrix, figsize=(n_matrix * 5, 5))

    if len(matrix_list) == 1:
        plot_heatmap(matrix_list[0])
    else:
        for i in range(n_matrix):
            matrix_i = matrix_list[i]
            # Plot the first matrix
            im_i = axes[i].imshow(matrix_i, cmap='viridis', origin='lower')
            axes[0].set_title(f'{title} {i}')

            # Add colorbars
            fig.colorbar(im_i, ax=axes[i])

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()



def plot_multiplayer_radar_chart(categories, players_data, player_names, title='Radar Chart'):
    num_players = len(players_data)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i in range(num_players):
        values = np.concatenate((players_data[i], [players_data[i][0]]))
        ax.fill(angles, values, alpha=0.25)
        ax.plot(angles, values, label=player_names[i], linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=15, y=1.1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig('radar_chart.png', bbox_inches='tight')
    plt.show()

# Example data for two players
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
player1_data = [4, 3, 5, 2, 4]
player2_data = [3, 4, 2, 5, 3]
player_names = ['Player 1', 'Player 2']

# Plot multiplayer radar chart
plot_multiplayer_radar_chart(categories, [player1_data, player2_data], player_names, title='Multiplayer Radar Chart')