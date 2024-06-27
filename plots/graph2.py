import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('Agg')

def plot_data(evaluate=False, lag=0, attacker=True):
    """
    Plots differences in valid and invalid actions for attacker and defender over training cycles.

    Parameters:
    evaluate (bool): If True, plots from 'out_eval.csv'. Otherwise, plots from 'out.csv'.
    """
    # File selection and configuration
    file_name = 'out_eval.csv' if evaluate else 'out.csv'
    max_step = 9 if evaluate else 19
    highlight_every = 2 if evaluate else 12
    highlight_span = 1 if evaluate else 6
    max_episodes = 20

    #max_step = 6
    #max_episodes = 20
    #highlight_every = 12
    #highlight_span = 6

    # Directory where data files are stored 
    data_dir = 'model_saves/ladder_20240618_131013_3_2'
    file_path = f'{data_dir}/{file_name}'

    # Read the data from the CSV file
    data = pd.read_csv(file_path)
    data = data.iloc[:max_step * max_episodes]  # Include only up to max_step
    print(data)

    # Define the columns for attacker and defender valid and invalid actions
    columns_to_plot = {
        "Attacker Valid Actions": "mean_attacker_valid",
        "Attacker Invalid Actions": "mean_attacker_invalid",
        "Defender Valid Actions": "mean_defender_valid",
        "Defender Invalid Actions": "mean_defender_invalid"
    }

    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Assign subplots to attackers on the left and defenders on the right
    subplot_mapping = {
        "Attacker Valid Actions": axs[0, 0],
        "Attacker Invalid Actions": axs[1, 0],
        "Defender Valid Actions": axs[0, 1],
        "Defender Invalid Actions": axs[1, 1]
    }

    # Initialize previous values for computing differences
    previous_values = {key: 0 for key in columns_to_plot.keys()}

    for title, col in columns_to_plot.items():
        ax = subplot_mapping[title]
        differences = []

        # Compute differences
        for i in range(len(data)):
            if i % highlight_every == 0 and i != 0:
                previous_values[title] = data[col].iloc[i-1]  # Update previous value at switch point
            differences.append((data[col].iloc[i] - previous_values[title]) / previous_values[title])

        # Plot differences
        ax.plot(data.index, differences, label=f'{col} (Difference)')
        ax.set_title(f'{title} Differences')
        ax.set_xlabel('Step')
        ax.set_ylabel('Difference Value')
        ax.legend()

        # Highlight sections based on training type
        for j in range(0, len(data.index), highlight_every):
            start_highlight = j
            end_highlight = min(j + highlight_span, len(data.index) - 1)
            ax.axvspan(start_highlight, end_highlight, color='blue', alpha=0.1)  # Attacker training
            if j + highlight_span < len(data.index):
                ax.axvspan(end_highlight, min(j + highlight_every, len(data.index) - 1), color='red', alpha=0.1)  # Defender training

    # Adjust layout and save plot
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)  # Ensure the plots directory exists
    plt.savefig(f'plots/{data_dir.split("/")[1]}_{"eval" if evaluate else "train"}_out_diff.png')

# Example of calling the function
plot_data(evaluate=False, lag=0, attacker=True)  # For training data