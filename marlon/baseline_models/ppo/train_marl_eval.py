from stable_baselines3 import PPO

from datetime import datetime
import os
import pandas as pd

from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe_delayed_single import MultiAgentUniverse

ENV_MAX_TIMESTEPS = 2000
LEARN_TIMESTEPS = 960000
LEARN_EPISODES = 1200000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
DEFENDER_INVALID_ACTION_REWARD = 0
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = False
EVALUATE_EPISODES = 5
ATTACKER_SAVE_PATH = 'ppo_marl_attacker.zip'
DEFENDER_SAVE_PATH = 'ppo_marl_defender.zip'


SAVE_DIR = f'model_saves/ladder_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(SAVE_DIR, exist_ok=True)


with open(f'{SAVE_DIR}/info.txt', 'w') as file:
    file.write(f"ENV_MAX_TIMESTEPS = {ENV_MAX_TIMESTEPS}\n")
    file.write(f"LEARN_TIMESTEPS = {LEARN_TIMESTEPS}\n")
    file.write(f"LEARN_EPISODES = {LEARN_EPISODES}\n")


def train(evaluate_after=False):
    universe = MultiAgentUniverse.build(
        env_id='CyberBattleToyCtf-v0',
        attacker_builder=BaselineAgentBuilder(
            alg_type=PPO,
            policy='MultiInputPolicy'
        ),
        defender_builder=BaselineAgentBuilder(
            alg_type=PPO,
            policy='MultiInputPolicy'
        ),
        attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
        attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
        defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN
    )

    universe.learn(
        total_timesteps=LEARN_TIMESTEPS,
        n_eval_episodes=LEARN_EPISODES
    )

    universe.save(
        attacker_filepath=f'{SAVE_DIR}/{ATTACKER_SAVE_PATH}',
        defender_filepath=f'{SAVE_DIR}/{DEFENDER_SAVE_PATH}'
    )

    if evaluate_after:
            stats, defender_actions = universe.evaluate(
                n_episodes=EVALUATE_EPISODES
            )

            columns = ["mean_length", "std_length", "mean_attacker_reward", "std_attacker_reward", 
                       "mean_attacker_valid", "std_attacker_valid", "mean_attacker_invalid", "std_attacker_invalid",
                       "mean_defender_reward", "std_defender_reward", "mean_defender_valid", "std_defender_valid", 
                       "mean_defender_invalid", "std_defender_invalid","mean_network_avilability",
                       "std_network_avilability","min_network_avilability", "mean_network_infected",
                       "std_network_infected"]
            
            out = [stats.mean_length, stats.std_length, stats.mean_attacker_reward, stats.std_attacker_reward,
            stats.mean_attacker_valid, stats.std_attacker_valid, stats.mean_attacker_invalid, stats.std_attacker_invalid,
            stats.mean_defender_reward, stats.std_defender_reward, stats.mean_defender_valid, stats.std_defender_valid, 
            stats.mean_defender_invalid, stats.std_defender_invalid, 
            stats.mean_network_availability, stats.std_network_availability, stats.min_network_availability, stats.mean_network_infected, stats.std_network_infected]
        
            pd.DataFrame(data=[out], columns=columns).to_csv(f'{SAVE_DIR}/out.csv', mode='a', header=not os.path.exists(f'{SAVE_DIR}/out.csv'), index=False)
        
            pd.DataFrame(data=defender_actions).to_csv(f'{SAVE_DIR}/0_actions.csv', header=False, index=False)

if __name__ == '__main__':
    train(evaluate_after=True)
