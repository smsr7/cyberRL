from stable_baselines3 import PPO

from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse

ENV_MAX_TIMESTEPS = 1500
LEARN_TIMESTEPS = 480_000
LEARN_EPISODES = 1200000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
DEFENDER_INVALID_ACTION_REWARD = -1
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = False
EVALUATE_EPISODES = 5
ATTACKER_SAVE_PATH = 'ppo_marl_attacker.zip'
DEFENDER_SAVE_PATH = 'ppo_marl_defender.zip'

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
        defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
    )

    universe.learn(
        total_timesteps=LEARN_TIMESTEPS,
        n_eval_episodes=LEARN_EPISODES
    )

    universe.save(
        attacker_filepath=ATTACKER_SAVE_PATH,
        defender_filepath=DEFENDER_SAVE_PATH
    )

    if evaluate_after:
        universe.evaluate(
            n_episodes=EVALUATE_EPISODES
        )

if __name__ == '__main__':
    train(evaluate_after=True)
