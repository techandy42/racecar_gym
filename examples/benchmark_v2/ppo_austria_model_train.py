import os
import shutil
import argparse
import gymnasium
import racecar_gym.envs.gym_api
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from huggingface_hub import login
from huggingface_sb3 import push_to_hub
from lidar_only_model import FlattenDictActionWrapper

parser = argparse.ArgumentParser(description='Train a PPO model for Austria track.')
parser.add_argument('--huggingface_token', type=str, help='HuggingFace CLI login token.')
parser.add_argument('--huggingface_username', type=str, help='HuggingFace account username.')
parser.add_argument('--total_timesteps', type=int, default=1e5, help='Total number of training timesteps.')
parser.add_argument('--render_mode', type=str, default='human', help='Render mode for the environment.')
parser.add_argument('--model_id', type=str, default='ppo_austria_model', help='Filename to save the trained model.')
args = parser.parse_args()

if args.huggingface_token is None:
    raise ValueError("Must provide a HuggingFace CLI login token if uploading model.")
elif args.huggingface_username is None:
    raise ValueError("Must provide a HuggingFace account username if uploading model.")
else:
    login(token=args.huggingface_token)

# Wrap the environment
env = FlattenDictActionWrapper(gymnasium.make('SingleAgentAustria-v0', render_mode=args.render_mode))

# Make the environment VecEnv compatible
env = make_vec_env(lambda: env)

# Initialize the PPO model with MultiInputPolicy
model = PPO("MultiInputPolicy", env, verbose=1)

print(f"Training PPO model for {args.total_timesteps} timesteps")

model.learn(total_timesteps=args.total_timesteps)

print("Finished training")
env.close()

model.save(args.model_id)

push_to_hub(
    repo_id=f"{args.huggingface_username}/{args.model_id}",
    filename=f"{args.model_id}.zip",
    commit_message="Initial commit",
)

# Step 1: Create a directory named "models" if it doesn't exist
directory_name = "models"
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# Step 2: Move a zip file from the current directory to the "models" directory
zip_file_name = f"{args.model_id}.zip"  # Replace with your zip file's name
destination = os.path.join(directory_name, zip_file_name)
shutil.move(zip_file_name, destination)