import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import math
import glob

matplotlib.use('Agg')

def plot_action_data(directory):
    """
    Process action data from multiple CSV files and return action frequencies.

    Parameters:
    directory (str): The directory where action CSV files are stored.
    """
    # Initialize a list to store action frequencies for each episode
    action_frequencies = []

    action_labels = {
        0: 'REIMAGE',
        1: 'Block Traffic',
        2: 'Allow Traffic',
        3: 'Stop Service',
        4: 'Start Service'
    }
    
    # Iterate through each CSV file
    for i in range(20):
        data = pd.read_csv(f'{directory}/{i}_actions.csv')
        
        # Count the frequency of each action
        action_counts = data.iloc[:, 0].value_counts(normalize=True) * 100
        
        # Store the action frequencies for the current episode
        action_frequencies.append(action_counts)

    # Convert the list of action frequencies to a DataFrame
    action_df = pd.DataFrame(action_frequencies).fillna(0)
    
    # Re-label the columns based on action_labels
    action_df = action_df.rename(columns=action_labels)

    return action_df

def plot_data(evaluate=False, lag=0, attacker=True):
    """
    Plots data from a CSV file and highlights training phases, including action frequencies.

    Parameters:
    evaluate (bool): If True, plots from 'eval_out.csv'. Otherwise, plots from 'out.csv'.
    """
    # File selection and configuration
    file_name = 'out_eval.csv' if evaluate else 'out.csv'
    max_step = 9 if evaluate else 19
    highlight_every = 2 if evaluate else 12
    highlight_span = 1 if evaluate else 6
    max_episodes = 20

    # Directory where data files are stored
    data_dir = 'model_saves/baseline_ladder_20240624_215853'
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
    fig, axs = plt.subplots(5, 1, figsize=(15, 25), gridspec_kw={'height_ratios': [1, 1, 1, 1, 2]})

    # Assign subplots to ensure attackers are on the left and defenders on the right
    subplot_mapping = {
        "Attacker Rewards": axs[0],
        "Attacker Valid Actions": axs[1],
        "Defender Rewards": axs[2],
        "Defender Valid Actions": axs[3],
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
    ax_infected = axs[4]
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
    ax_availability = ax_infected.twinx()
    ax_availability.plot(data.index, data['mean_network_avilability'], label='Mean Network Availability', color='blue')
    ax_availability.fill_between(data.index, 
                                 data['mean_network_avilability'] - data['std_network_avilability'], 
                                 data['mean_network_avilability'] + data['std_network_avilability'], 
                                 color='blue', alpha=0.2)
    ax_availability.plot(data.index, data['min_network_avilability'], label='Min Network Availability', color='red')
    ax_availability.legend(loc='upper right')

    # Plot action frequencies
    action_df = plot_action_data('model_saves/delay_ladder_20240624_214924')
    ax_actions = plt.axes([0.1, -0.3, 0.8, 0.2])
    action_df.plot(kind='bar', stacked=True, ax=ax_actions, width=1.0)
    ax_actions.set_title('Action Frequencies Over Episodes')
    ax_actions.set_xlabel('Episode')
    ax_actions.set_ylabel('Action Frequency (%)')
    ax_actions.set_xticks(range(20))
    ax_actions.set_xticklabels(range(20))
    ax_actions.legend(title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout and save plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs('plots', exist_ok=True)  # Ensure the plots directory exists
    plt.savefig(f'plots/{data_dir.split("/")[1]}_{"eval" if evaluate else "train"}_combined.png')

# Example of calling the function
plot_data(evaluate=False, lag=0, attacker=True)  # For training data
# plot_data(evaluate=True)   # For evaluation data
