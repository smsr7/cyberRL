import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def process_action_data(directory):
    """
    Process action data from multiple CSV files and plot action frequencies.

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

    # Colors for each action
    action_colors = {
        'REIMAGE': 'purple',
        'Block Traffic': 'red',
        'Allow Traffic': 'dodgerblue',
        'Stop Service': 'orange',
        'Start Service': 'green'
    }

    # Iterate through each CSV file
    for i in range(20):
        data = pd.read_csv(f'model_saves/{directory}/{i}_actions.csv')
        
        # Count the frequency of each action
        action_counts = data.iloc[:, 0].value_counts(normalize=True) * 100
        
        # Store the action frequencies for the current episode
        action_frequencies.append(action_counts)

    # Convert the list of action frequencies to a DataFrame
    action_df = pd.DataFrame(action_frequencies).fillna(0)
    
    # Re-label the columns based on action_labels
    action_df = action_df.rename(columns=action_labels)
    
    # Plot the action frequencies as bar graphs
    fig, ax = plt.subplots(figsize=(15, 8))
    action_df.plot(kind='bar', stacked=True, ax=ax, width=1.0, color=[action_colors[col] for col in action_df.columns], alpha=0.6)

    # Customize the plot
    ax.set_title('Action Frequencies Over Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Action Frequency (%)')
    ax.legend(title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks(range(20))  # Set x-ticks to be 0-19
    ax.set_xticklabels(range(20))  # Label x-ticks to be 0-19

    # Adjust layout and save plot
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{directory}_action_frequencies.png')

# Example of calling the function
process_action_data('ladder_20240626_210722')
