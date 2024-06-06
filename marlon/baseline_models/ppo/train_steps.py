from stable_baselines3 import PPO
import pandas as pd
import os

from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder

ENV_MAX_TIMESTEPS = 1500
LEARN_TIMESTEPS = 10000
LEARN_EPISODES = 10000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
EVALUATE_EPISODES = 5
SAVE_DIR = 'model_saves/baseline'
ATTACKER_SAVE_PATH = 'ppo_attacker'

def train(step=0, max_steps=10, evaluate_after=False):
    if step < max_steps:
        if step == 0:
            universe = MultiAgentUniverse.build(
            env_id='CyberBattleToyCtf-v0',
            attacker_builder=RandomAgentBuilder(),
            defender_builder=BaselineAgentBuilder(
                alg_type=PPO,
                policy='MultiInputPolicy'
            ),
            attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD,
            defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
            defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
            )
        else:
            universe = MultiAgentUniverse.build(
                env_id='CyberBattleToyCtf-v0',
                attacker_builder=LoadFileBaselineAgentBuilder(
                        alg_type=PPO,
                        file_path=f'{SAVE_DIR}/{step-1}_{ATTACKER_SAVE_PATH}'
                        ),
                attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
                attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
                )
        universe.learn(
            total_timesteps=LEARN_TIMESTEPS,
            n_eval_episodes=LEARN_EPISODES
        )

        universe.save(
            attacker_filepath=f'{SAVE_DIR}/{step}_{ATTACKER_SAVE_PATH}'
        )

        if evaluate_after:
            stats = universe.evaluate(
                n_episodes=EVALUATE_EPISODES
            )

            columns = ["step", "mean_length", "std_length", "mean_attacker_reward", "std_attacker_reward", 
                       "mean_attacker_valid", "std_attacker_valid", "mean_attacker_invalid", "std_attacker_invalid",
                        "mean_defender_reward", "std_defender_reward", "mean_defender_valid", "std_defender_valid", 
                        "mean_defender_invalid", "std_defender_invalid"]
            out = [step, stats.mean_length, stats.std_length, stats.mean_attacker_reward, stats.std_attacker_reward,
            stats.mean_attacker_valid, stats.std_attacker_valid, stats.mean_attacker_invalid, stats.std_attacker_invalid,
            stats.mean_defender_reward, stats.std_defender_reward, stats.mean_defender_valid, stats.std_defender_valid, 
            stats.mean_defender_invalid, stats.std_defender_invalid]
        
            pd.DataFrame(data=[out], columns=columns).to_csv(f'{SAVE_DIR}/out_attacker.csv', mode='a', header=not os.path.exists(f'{SAVE_DIR}/out.csv'), index=False)
            
        train(step=step+1, evaluate_after=True, max_steps=10)

if __name__ == '__main__':
    train(evaluate_after=True)
