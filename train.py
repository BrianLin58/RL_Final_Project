import time
import warnings
import argparse
import gymnasium as gym
from gymnasium.envs.registration import register

import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
import numpy as np

class Map3DCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Map3DCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(2, 2, 2), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).unsqueeze(1).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.unsqueeze(1)))

warnings.filterwarnings("ignore")
register(
    id='VideoEnv-v0',
    entry_point='envs:VideoEnv'
)

# Set hyper params (configurations) for training
# my_config = {
#     "run_id": "example",
#     "algorithm": SAC,
#     "policy_network": "CnnPolicy",
#     "save_path": "models\\sample_model",
#     "num_train_envs": 4,
#     "epoch_num": 5,
#     "buffer_size": 100000,
#     "timesteps_per_epoch": 100,
#     "eval_episode_num": 10
# }

my_config = {
    "run_id": "video_enhancement",
    "algorithm": PPO,  # PPO works well with discrete actions
    "policy_network": "MlpPolicy", # use MlpPolicy for non-image obs (encoder output)
    "save_path": "models\\video_enhancement_model",
    "num_train_envs": 1,
    "epoch_num": 1,
    "timesteps_per_epoch": 1,
    "eval_episode_num": 1,
    "batch_size": 2,
    "n_steps": 2
}

def make_env():
    env = gym.make('VideoEnv-v0')
    env = Monitor(env)
    return env

def eval(env, model, eval_episode_num, visualize_index):
    """Evaluate the model and return avg_reward"""
    total_reward = 0.0

    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()
        ep_reward = 0.0

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # Handle vectorized environment
            if isinstance(reward, (list, np.ndarray)):
                ep_reward += reward[0]
            else:
                ep_reward += reward

        total_reward += ep_reward
        if seed in visualize_index:
            # env.envs[0].env.visualize(seed)
            env.env_method("visualize", seed)

    return total_reward / eval_episode_num

def train(eval_env, model, config, args):
    """Train agent using SB3 algorithm and my_config"""
    
    print(f"\n{'='*60}")
    print(f"Training Start")
    print(f"{'='*60}")

    print(f"\nModels will be saved to: {config['save_path']}\n")

    best_reward = -100

    start_time = time.time()

    for epoch in range(config["epoch_num"]):
        epoch_start_time = time.time()

        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
        )

        epoch_duration = time.time() - epoch_start_time

        # Evaluation
        print("[DEBUG] Start evaluation...")
        eval_start = time.time()
        avg_reward = eval(eval_env, model, config["eval_episode_num"], args.visualize_index)
        eval_duration = time.time() - eval_start

        total_duration = time.time() - start_time

        # print training progress and speed
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epoch_num']} completed")
        print(f"{'='*60}")
        print(f"Training Speed:")
        print(f"   - Epoch time: {epoch_duration:.1f}s")
        print(f"   - Eval time:  {eval_duration:.1f}s")
        print(f"   - Total time: {total_duration/60:.1f} min")
        print(f"Performance:")
        print(f"   - Avg Reward: {avg_reward:.4f}")

        # Save the best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            print("Saving best model...")
            model.save(f"{config['save_path']}/{config['algorithm'].__name__}")

        print("-" * 60)

    total_time = (time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize_index", nargs = "+", type = int, default = [], help = "Visualize at the specified round of evaluation.")
    args = parser.parse_args()

    train_env = SubprocVecEnv([make_env for _ in range(my_config["num_train_envs"])])

    eval_env = DummyVecEnv([make_env])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

    if my_config["policy_network"] == "CnnPolicy":
        policy_kwargs = dict(
            features_extractor_class=Map3DCNN,
            net_arch=[256, 256]
        )
    else:
        policy_kwargs = dict(
            net_arch=[256, 256]
        )
    
    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    # model = my_config["algorithm"](
    #     my_config["policy_network"], 
    #     train_env, 
    #     verbose=1,
    #     device=device,
    #     tensorboard_log=my_config["run_id"],
    #     buffer_size=my_config["buffer_size"],
    #     policy_kwargs=policy_kwargs
    # )
    model = my_config["algorithm"](
            my_config["policy_network"], 
            train_env, 
            verbose=1,
            device=device,
            tensorboard_log=my_config["run_id"],
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=my_config["n_steps"],
            batch_size=my_config["batch_size"],
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
    train(eval_env, model, my_config, args)
