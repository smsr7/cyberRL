import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import math


matplotlib.use('Agg')

def plot_data(evaluate=False, lag=0, attacker=True):
    """
    Plots data from a CSV file and highlights training phases.

    Parameters:
    evaluate (bool): If True, plots from 'eval_out.csv'. Otherwise, plots from 'out.csv'.
    """
    # File selection and configuration
    file_name = 'out_eval.csv' if evaluate else 'out.csv'
    max_step = 6
    max_episodes = 20
    highlight_every = 12
    highlight_span = 6

    # Directory where data files are stored
    data_dir = 'model_saves/ladder_20240612_032539'
    file_path = f'{data_dir}/{file_name}'

    # Read the data from the CSV file
    data = pd.read_csv(file_path)
    data = data.iloc[:max_step * max_episodes]  # Include only up to max_step * max_episodes
    print(data)

    # Define the columns for attacker and defender rewards and actions
    columns_to_plot = {
        "Attacker Rewards": ("mean_attacker_reward", "std_attacker_reward"),
        "Attacker Valid Actions": ("mean_attacker_valid", "std_attacker_valid"),
        "Attacker Invalid Actions": ("mean_attacker_invalid", "std_attacker_invalid"),
        "Defender Rewards": ("mean_defender_reward", "std_defender_reward"),
        "Defender Valid Actions": ("mean_defender_valid", "std_defender_valid"),
        "Defender Invalid Actions": ("mean_defender_invalid", "std_defender_invalid")
    }

    # Create plots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    # Assign subplots to ensure attackers are on the left and defenders on the right
    subplot_mapping = {
        "Attacker Rewards": axs[0, 0],
        "Attacker Valid Actions": axs[1, 0],
        "Attacker Invalid Actions": axs[2, 0],
        "Defender Rewards": axs[0, 1],
        "Defender Valid Actions": axs[1, 1],
        "Defender Invalid Actions": axs[2, 1]
    }

    for title, (mean_col, std_col) in columns_to_plot.items():
        ax = subplot_mapping[title]
        # Shift x-tick by 1
        x_vals = data.index + 2
        ax.plot(x_vals, data[mean_col], label=f'{mean_col}')
        ax.fill_between(x_vals, data[mean_col] - data[std_col], data[mean_col] + data[std_col], alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()

        # Highlight sections based on training type
        for j in range(0, max_step * max_episodes, highlight_every):
            start_highlight = j + 1
            end_highlight = min(j + highlight_span + 1, max_step * max_episodes + 1)
            ax.axvspan(start_highlight, end_highlight, color='blue', alpha=0.1)  # Attacker training
            if j + highlight_span < max_step * max_episodes:
                ax.axvspan(end_highlight, min(j + highlight_every + 1, max_step * max_episodes + 1), color='red', alpha=0.1)
                if lag and attacker:
                    ax.axvspan(end_highlight, end_highlight + math.floor(highlight_span * lag), color='red', alpha=0.1)

                
                
    # Adjust layout and save plot
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)  # Ensure the plots directory exists
    plt.savefig(f'plots/{data_dir.split("/")[1]}_{"eval" if evaluate else "train"}_out.png')

# Example of calling the function
plot_data(evaluate=False, lag=0.6, attacker=True)  # For training data
#plot_data(evaluate=True)   # For evaluation data
