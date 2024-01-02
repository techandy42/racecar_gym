import os
import shutil
import argparse
import torch as th
import torch.nn as nn
from typing import Dict
import numpy as np
import gymnasium
import racecar_gym.envs.gym_api
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from huggingface_hub import login
from huggingface_sb3 import push_to_hub

parser = argparse.ArgumentParser(description='Train a PPO model for Austria track.')
parser.add_argument('--upload', type=bool, default=False, help='Upload model to HuggingFace if True.')
parser.add_argument('--huggingface_token', type=str, help='HuggingFace CLI login token.')
parser.add_argument('--huggingface_username', type=str, help='HuggingFace account username.')
parser.add_argument('--total_timesteps', type=int, default=1e5, help='Total number of training timesteps.')
parser.add_argument('--render_mode', type=str, default='human', help='Render mode for the environment.')
parser.add_argument('--model_save_name', type=str, default='ppo_austria_model', help='Filename to save the trained model.')
args = parser.parse_args()

if args.upload:
    if (args.huggingface_token is None) or (args.huggingface_username is None):
        raise ValueError("Must provide a HuggingFace CLI login token and account username if uploading model.")
    else:
        login(token=args.huggingface_token)

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # CNN for lidar data
        # Assuming LiDAR data is a single-channel input (1, N)
        n_lidar_points = observation_space['lidar'].shape[0]
        self.cnn_lidar = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_lidar_data = th.as_tensor(observation_space['lidar'].sample()[None, None, :]).float()
            n_flatten_lidar = self.cnn_lidar(sample_lidar_data).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten_lidar, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Reshape LiDAR data to have dimensions [batch_size, channels, length]
        # where channels is 1 and length is the number of LiDAR points.
        lidar_data = observations['lidar']
        if len(lidar_data.shape) == 1:
            # Add batch and channel dimensions if they are not present
            lidar_data = lidar_data.unsqueeze(0).unsqueeze(0)
        elif len(lidar_data.shape) == 2:
            # Add channel dimension if only batch dimension is present
            lidar_data = lidar_data.unsqueeze(1)

        lidar_processed = self.cnn_lidar(lidar_data)
        return self.linear(lidar_processed)

# Custom wrapper to flatten the action space
class FlattenDictActionWrapper(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Dict({
            'pose': env.observation_space['pose'],
            'velocity': env.observation_space['velocity'],
            'acceleration': env.observation_space['acceleration'],
            'lidar': env.observation_space['lidar']
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
            'acceleration': observation['acceleration'],
            'lidar': observation['lidar']
        })
        return filtered_observation

# Wrap the environment
env = FlattenDictActionWrapper(gymnasium.make('SingleAgentAustria-v0', render_mode=args.render_mode))

# Make the environment VecEnv compatible
env = make_vec_env(lambda: env)

# Customize the policy network
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256)  # Adjust as needed
)

# Initialize the PPO model with MultiInputPolicy
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

args.total_timesteps = 1 # Replace with the number of timesteps you want to train for
print("Training PPO model for {} timesteps".format(args.total_timesteps))

model.learn(total_timesteps=args.total_timesteps)

print("Finished training")
env.close()

model.save(args.model_save_name)

if args.upload:
    push_to_hub(
        repo_id=f"{args.huggingface_username}/{args.model_save_name}",
        filename=f"{args.model_save_name}.zip",
        commit_message="Initial commit",
    )

# Step 1: Create a directory named "models" if it doesn't exist
directory_name = "models"
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# Step 2: Move a zip file from the current directory to the "models" directory
zip_file_name = f"{args.model_save_name}.zip"  # Replace with your zip file's name
destination = os.path.join(directory_name, zip_file_name)
shutil.move(zip_file_name, destination)