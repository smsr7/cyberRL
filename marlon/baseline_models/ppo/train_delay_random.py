from stable_baselines3 import PPO
from datetime import datetime
import os
import pandas as pd

from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from marlon.baseline_models.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe_delayed_random import MultiAgentUniverse
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder

ENV_MAX_TIMESTEPS = 5000
LEARN_TIMESTEPS = 4000
LEARN_EPISODES = 10000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
DEFENDER_INVALID_ACTION_REWARD = 0
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = False
EVALUATE_EPISODES = 5

SAVE_DIR = f'model_saves/ladder_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
ATTACKER_SAVE_PATH = 'ppo_attacker'
DEFENDER_SAVE_PATH = 'ppo_defender'

os.makedirs(SAVE_DIR, exist_ok=True)


with open(f'{SAVE_DIR}/info.txt', 'w') as file:
    file.write(f"ENV_MAX_TIMESTEPS = {ENV_MAX_TIMESTEPS}\n")
    file.write(f"LEARN_TIMESTEPS = {LEARN_TIMESTEPS}\n")
    file.write(f"LEARN_EPISODES = {LEARN_EPISODES}\n")

    
    
def train(episode=0, step=0, evaluate_after=True, max_episodes=3, max_steps=2, defense_start=True, lag=False, delay=0):
    directory = SAVE_DIR if not delay else f'{SAVE_DIR}/{delay}'
    if delay:
        os.makedirs(directory, exist_ok=True)

    print(step, episode, max_steps, max_episodes)
    
    if episode < max_episodes:
        if (defense_start) and (episode == 0):
            if step == 0:
                universe = MultiAgentUniverse.build(
                env_id='CyberBattleToyCtf-v0',
                attacker_builder=RandomAgentBuilder(),
                defender_builder=BaselineAgentBuilder(
                    alg_type=PPO,
                    policy='MultiInputPolicy'
                ),
                defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
                defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
                )
            else:
                universe = MultiAgentUniverse.build(
                env_id='CyberBattleToyCtf-v0',
                attacker_builder=RandomAgentBuilder(),
                defender_builder=LoadFileBaselineAgentBuilder(
                        alg_type=PPO,
                        file_path=f'{directory}/{episode}_{step-1}_{DEFENDER_SAVE_PATH}'
                        ),
                attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
                attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
                defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
                defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
                )
        else: 
            if episode == 1:
                if step == 0:
                    universe = MultiAgentUniverse.build(
                    env_id='CyberBattleToyCtf-v0',
                    attacker_builder=BaselineAgentBuilder(
                        alg_type=PPO,
                        policy='MultiInputPolicy'
                    ),
                    defender_builder=LoadFileBaselineAgentBuilder(
                    alg_type=PPO,
                    file_path=f'{directory}/{episode-1}_{max_steps}_{DEFENDER_SAVE_PATH}'
                    ),
                    attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
                    attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
                    defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
                    defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
                    )
                else:
                    universe = MultiAgentUniverse.build(
                    env_id='CyberBattleToyCtf-v0',
                    attacker_builder=LoadFileBaselineAgentBuilder(
                            alg_type=PPO,
                            file_path=f'{directory}/{episode}_{step-1}_{ATTACKER_SAVE_PATH}'
                            ),
                    defender_builder=LoadFileBaselineAgentBuilder(
                            alg_type=PPO,
                            file_path=f'{directory}/{episode-1}_{max_steps}_{DEFENDER_SAVE_PATH}'
                            ),
                    attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
                    attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
                    defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
                    defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
                    )
            else:
                if (episode%2 == 1):
                    universe = MultiAgentUniverse.build(
                    env_id='CyberBattleToyCtf-v0',
                    attacker_builder=LoadFileBaselineAgentBuilder(
                            alg_type=PPO,
                            file_path=f'{directory}/{episode if step != 0 else episode-2}_{step-1 if step != 0 else max_steps}_{ATTACKER_SAVE_PATH}'
                            ),
                    defender_builder=LoadFileBaselineAgentBuilder(
                            alg_type=PPO,
                            file_path=f'{directory}/{episode-1}_{max_steps}_{DEFENDER_SAVE_PATH}'
                            ), 
                    attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
                    attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
                    defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
                    defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
                    )
                else:
                    universe = MultiAgentUniverse.build(
                    env_id='CyberBattleToyCtf-v0',
                    attacker_builder=LoadFileBaselineAgentBuilder(
                            alg_type=PPO,
                            file_path=f'{directory}/{episode-1}_{max_steps}_{ATTACKER_SAVE_PATH}'
                            ),
                    defender_builder=LoadFileBaselineAgentBuilder(
                            alg_type=PPO,
                            file_path=f'{directory}/{episode if step != 0 else episode-2}_{step-1 if step != 0 else max_steps}_{DEFENDER_SAVE_PATH}'
                            ), 
                    attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
                    attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
                    defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
                    defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
                    )
            
        universe.learn(
                total_timesteps=LEARN_TIMESTEPS if ((episode % 2) == 0) else int(LEARN_TIMESTEPS*1.5),
                n_eval_episodes=LEARN_EPISODES,
                ladder=True,
                attacker_train=((episode % 2) == 1)

            )

        if ((episode%2) == 0):
            universe.save(
                defender_filepath=f'{directory}/{episode}_{step}_{DEFENDER_SAVE_PATH}'
                )
        else:
            universe.save(
                   attacker_filepath=f'{directory}/{episode}_{step}_{ATTACKER_SAVE_PATH}',        
                )

        if evaluate_after:
            stats, defender_actions = universe.evaluate(
                n_episodes=EVALUATE_EPISODES
            )

            columns = ["step", "episode", "mean_length", "std_length", "mean_attacker_reward", "std_attacker_reward", 
                       "mean_attacker_valid", "std_attacker_valid", "mean_attacker_invalid", "std_attacker_invalid",
                       "mean_defender_reward", "std_defender_reward", "mean_defender_valid", "std_defender_valid", 
                       "mean_defender_invalid", "std_defender_invalid","mean_network_avilability",
                       "std_network_avilability","min_network_avilability", "mean_network_infected",
                       "std_network_infected"]
            
            out = [step, episode, stats.mean_length, stats.std_length, stats.mean_attacker_reward, stats.std_attacker_reward,
            stats.mean_attacker_valid, stats.std_attacker_valid, stats.mean_attacker_invalid, stats.std_attacker_invalid,
            stats.mean_defender_reward, stats.std_defender_reward, stats.mean_defender_valid, stats.std_defender_valid, 
            stats.mean_defender_invalid, stats.std_defender_invalid, 
            stats.mean_network_availability, stats.std_network_availability, stats.min_network_availability, stats.mean_network_infected, stats.std_network_infected]
        
            pd.DataFrame(data=[out], columns=columns).to_csv(f'{directory}/out.csv', mode='a', header=not os.path.exists(f'{directory}/out.csv'), index=False)
            
            pd.DataFrame(data=defender_actions).to_csv(f'{directory}/{episode}_actions.csv', header=False, index=False)
        
        
        if step<max_steps:
            train(episode=episode, step=step+1, evaluate_after=True, max_episodes=max_episodes, max_steps=max_steps)
        else:
            train(episode=episode+1, step=0, evaluate_after=True, max_episodes=max_episodes, max_steps=max_steps)

if __name__ == '__main__':
    train(max_episodes=20, max_steps=5, defense_start=True, lag=False)
