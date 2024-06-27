import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import math

matplotlib.use('Agg')

def plot_data(evaluate=False, lag=0, attacker=False):
    """
    Plots data from a CSV file and highlights training phases.

    Parameters:
    evaluate (bool): If True, plots from 'eval_out.csv'. Otherwise, plots from 'out.csv'.
    """
    # File selection and configuration
    file_name = 'out_eval.csv' if evaluate else 'out.csv'
    max_step = 9 if evaluate else 19
    highlight_every = 12 if evaluate else 12
    highlight_span = 6 if evaluate else 6
    max_episodes = 10

    # Directory where data files are stored
    data_dir = 'model_saves/ladder_20240626_210722'
    file_path = f'{data_dir}/{file_name}'

    # Read the data from the CSV file
    data = pd.read_csv(file_path)
    data = data.iloc[:max_step * max_episodes]  # Include only up to max_step * max_episodes
    print(data)

    # Define the columns for attacker and defender rewards and actions
    columns_to_plot = {
        "Attacker Rewards": ("mean_attacker_reward", "std_attacker_reward"),
        "Attacker Valid Actions": ("mean_attacker_valid", "std_attacker_valid"),
        "Defender Rewards": ("mean_defender_reward", "std_defender_reward"),
        "Defender Valid Actions": ("mean_defender_valid", "std_defender_valid")
    }

    # Create plots
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))

    # Assign subplots to ensure attackers are on the left and defenders on the right
    subplot_mapping = {
        "Attacker Rewards": axs[0, 0],
        "Attacker Valid Actions": axs[1, 0],
        "Defender Rewards": axs[0, 1],
        "Defender Valid Actions": axs[1, 1],
    }

    for title, (mean_col, std_col) in columns_to_plot.items():
        ax = subplot_mapping[title]
        # Shift x-tick by 1
        x_vals = data.index + 1
        ax.plot(x_vals, data[mean_col], label=f'{mean_col}')
        ax.fill_between(x_vals, data[mean_col] - data[std_col], data[mean_col] + data[std_col], alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()

        # Highlight sections based on training type
        for j in range(0, len(data.index), highlight_every):
            start_highlight = j
            end_highlight = min(j + highlight_span, len(data.index))
            ax.axvspan(start_highlight, end_highlight, color='blue', alpha=0.1)  # Attacker training
            if j + highlight_span < len(data.index):
                ax.axvspan(end_highlight, min(j + highlight_every, len(data.index)), color='red', alpha=0.1)  # Defender training

    # Plot network infected metrics
    ax_infected = axs[2, 0]
    ax_infected.plot(data.index, data['mean_network_infected'], label='Mean Network Infected', color='green')
    ax_infected.fill_between(data.index, 
                             data['mean_network_infected'] - data['std_network_infected'], 
                             data['mean_network_infected'] + data['std_network_infected'], 
                             color='green', alpha=0.2)
    ax_infected.set_title('Network Infected Metrics')
    ax_infected.set_xlabel('Episode')
    ax_infected.set_ylabel('Value')
    ax_infected.set_ylim(0,0.4)
    ax_infected.legend()

    # Highlight sections based on training type for network infected plot
    for j in range(0, len(data.index), highlight_every):
        start_highlight = j
        end_highlight = min(j + highlight_span, len(data.index) - 1)
        ax_infected.axvspan(start_highlight, end_highlight, color='blue', alpha=0.1)  # Attacker training
        if j + highlight_span < len(data.index):
            ax_infected.axvspan(end_highlight, min(j + highlight_every, len(data.index) - 1), color='red', alpha=0.1)  # Defender training

    # Plot network availability metrics
    ax_availability = axs[2, 1]
    ax_availability.plot(data.index, data['mean_network_avilability'], label='Mean Network Availability', color='blue')
    ax_availability.fill_between(data.index, 
                                 data['mean_network_avilability'] - data['std_network_avilability'], 
                                 data['mean_network_avilability'] + data['std_network_avilability'], 
                                 color='blue', alpha=0.2)
    ax_availability.plot(data.index, data['min_network_avilability'], label='Min Network Availability', color='red')
    ax_availability.set_title('Network Availability Metrics')
    ax_availability.set_xlabel('Episode')
    ax_availability.set_ylabel('Value')
    ax_availability.legend()

    # Highlight sections based on training type for network availability plot
    for j in range(0, len(data.index), highlight_every):
        start_highlight = j
        end_highlight = min(j + highlight_span, len(data.index) - 1)
        ax_availability.axvspan(start_highlight, end_highlight, color='blue', alpha=0.1)  # Attacker training
        if j + highlight_span < len(data.index):
            ax_availability.axvspan(end_highlight, min(j + highlight_every, len(data.index) - 1), color='red', alpha=0.1)  # Defender training

    # Add a placeholder for the empty subplot
    fig.delaxes(axs[3, 0])
    fig.delaxes(axs[3, 1])

    # Adjust layout and save plot
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)  # Ensure the plots directory exists
    plt.savefig(f'plots/{data_dir.split("/")[1]}_{"eval" if evaluate else "train"}_out.png')

# Example of calling the function
plot_data(evaluate=False, lag=0, attacker=True)  # For training data
#plot_data(evaluate=True)   # For evaluation data
