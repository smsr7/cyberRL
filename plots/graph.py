import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

DATA_DIR = 'model_saves/ladder_20240605_131621'

# Read the data from the CSV file
data = pd.read_csv(f'{DATA_DIR}/out.csv')
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
    ax.plot(data.index, data[mean_col], label=f'{mean_col}')
    ax.fill_between(data.index, data[mean_col] - data[std_col], data[mean_col] + data[std_col], alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()

    # Highlight sections based on training type
    for j in range(0, len(data.index), 4):
        ax.axvspan(j, j + 2, color='blue', alpha=0.1)  # Attacker training
        ax.axvspan(j + 2, j + 4, color='red', alpha=0.1)  # Defender training

# Adjust layout and save plot
plt.tight_layout()
plt.savefig(f'plots/{DATA_DIR.split("/")[1]}_out.png')
