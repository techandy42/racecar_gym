from typing import Dict
import numpy as np
import gymnasium
import racecar_gym.envs.gym_api
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Custom wrapper to flatten the action space
class FlattenDictActionWrapper(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Dict({
            'pose': env.observation_space['pose'],
            'velocity': env.observation_space['velocity'],
            'acceleration': env.observation_space['acceleration']
        })
        self.action_space = gymnasium.spaces.Box(
            low=np.concatenate([space.low for space in env.action_space.spaces.values()]),
            high=np.concatenate([space.high for space in env.action_space.spaces.values()]),
            dtype=np.float32
        )

    def action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        motor, steering = np.split(action, [1])
        print(f"motor: {motor}, steering: {steering}")
        return {'motor': motor, 'steering': steering}

    def observation(self, observation):
        # Filter out unwanted observations
        filtered_observation = gymnasium.spaces.Dict({
            'pose': observation['pose'],
            'velocity': observation['velocity'],
            'acceleration': observation['acceleration']
        })
        return filtered_observation

# Wrap the environment
env = FlattenDictActionWrapper(gymnasium.make('SingleAgentAustria-v0', render_mode='human'))

# Make the environment VecEnv compatible
env = make_vec_env(lambda: env)

# Initialize the PPO model with MultiInputPolicy
model = PPO("MultiInputPolicy", env, verbose=1)

total_timesteps = 1 # Replace with the number of timesteps you want to train for
print("Training PPO model for {} timesteps".format(total_timesteps))
model.learn(total_timesteps=total_timesteps)

print("Finished training")
env.close()
model.save("ppo_racecar_model")
