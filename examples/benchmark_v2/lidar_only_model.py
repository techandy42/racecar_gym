import torch as th
import torch.nn as nn
from typing import Dict
import numpy as np
import gymnasium
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
        return {
            'pose': observation['pose'],
            'velocity': observation['velocity'],
            'acceleration': observation['acceleration'],
            'lidar': observation['lidar']
        }