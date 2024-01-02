import argparse
import gymnasium
import racecar_gym.envs.gym_api
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
from lidar_only_model import FlattenDictActionWrapper

# Parsing arguments
parser = argparse.ArgumentParser(description='Test a PPO model for Austria track.')
parser.add_argument('--huggingface_username', type=str, help='HuggingFace account username.')
parser.add_argument('--model_id', type=str, help='HuggingFace model ID.')
parser.add_argument('--max_steps', type=int, default=1e6, help='Total number of test timesteps.')
parser.add_argument('--render_mode', type=str, default='human', help='Render mode for the environment.')
args = parser.parse_args()

# Validating arguments
if args.huggingface_username is None:
    raise ValueError("Must provide a HuggingFace account username.")
elif args.model_id is None:
    raise ValueError("Must provide a HuggingFace model ID.")

# Loading the model from HuggingFace
checkpoint = load_from_hub(
    repo_id=f"{args.huggingface_username}/{args.model_id}",
    filename=f"{args.model_id}.zip",
)

model = PPO.load(checkpoint)

# Setting up the environment
test_env = FlattenDictActionWrapper(gymnasium.make('SingleAgentAustria-v0', render_mode=args.render_mode))
done = False
obs, state = test_env.reset(options=dict(mode='grid'))
total_rewards = 0

print(f"Testing {args.huggingface_username}/{args.model_id} model for maximum {args.max_steps} steps")

# Running the environment with the loaded model
t = 0
while not done:
    if t >= args.max_steps:
        break
    # Make sure the observation is in the correct format
    formatted_obs = test_env.observation(obs)
    action, _states = model.predict(formatted_obs, deterministic=True)
    obs, rewards, terminated, truncated, states = test_env.step(action)
    total_rewards += rewards
    done = terminated or truncated
    test_env.render()
    t += 1

print("=" * 50)
print("Total Steps:", t)
print("Total Reward:", total_rewards)
print("=" * 50)

# Closing the environment
test_env.close()
