import time
import warnings
import argparse
import yaml
import gymnasium as gym
from gymnasium.envs.registration import register

import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
import numpy as np
from envs.env import VideoEnv

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
# register(
#     id='VideoEnv-v0',
#     entry_point='envs:VideoEnv'
# )

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

# my_config = {
#     "run_id": "video_enhancement",
#     "algorithm": PPO,  # PPO works well with discrete actions
#     "policy_network": "MlpPolicy", # use MlpPolicy for non-image obs (encoder output)
#     "save_path": "test_vis",#"models/test_00",
#     "num_train_envs": 1,#4,
#     "epoch_num": 1,#100,
#     "timesteps_per_epoch": 3,#4096,
#     "eval_episode_num": 2,
#     "batch_size": 4,
#     "n_steps": 2#2048
# }

# def make_env():
#     env = gym.make('VideoEnv-v0')
#     env = Monitor(env)
#     return env

def make_env(cfg):
    def _init():
        return VideoEnv(
            data_dir = cfg["train"]["data_root"],
            val_dir = cfg["valid"]["data_root"],
            stack = cfg["train"]["stack"],
            action_repeat = cfg["train"]["action_repeat"],
            encoder = cfg["train"]["encoder"]
        )
    return _init

def eval(env, model, eval_episode_num, visualize_index):
    """Evaluate the model and return avg_reward"""
    total_reward = 0.0

    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        print(f"[DEBUG] Seed {seed} is in visualize_index {visualize_index}? {seed in visualize_index}")
        if seed in visualize_index:
            env.set_options([{"eval_id": seed, "vis_flag": True}])
        else:
            env.set_options([{"eval_id": seed}])
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
        # if seed in visualize_index:
            # env.envs[0].env.visualize(seed)
            # env.env_method("visualize", seed)

    return total_reward / eval_episode_num

def train(eval_env, model, cfg):
    """Train agent using SB3 algorithm and my_config"""
    config = cfg["train"]
    print(f"\n{'='*60}")
    print(f"Training Start")
    print(f"{'='*60}")

    print(f"\nModels will be saved to: {config['save_path']}\n")

    best_reward = -100

    start_time = time.time()

    for epoch in range(config["epoch_num"]):
        epoch_start_time = time.time()

        model.learn(
            total_timesteps = config["timesteps_per_epoch"],
            reset_num_timesteps = False,
            callback= WandbCallback(
                gradient_save_freq=100,
                verbose=2,
            ) if cfg["wandb"]["enabled"] else None,
        )

        epoch_duration = time.time() - epoch_start_time

        # Evaluation
        print("[DEBUG] Start evaluation...")
        eval_start = time.time()
        avg_reward = eval(eval_env, model, config["eval_episode_num"], cfg["visualize"]["index"])
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

        if cfg["wandb"]["enabled"]:
            wandb.log(
                {
                 "epoch": epoch,
                 "avg_reward": avg_reward,
                #  "avg_highest": avg_highest,
                #  "avg_score": avg_score
                 }
            )

        # Save the best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            print("Saving best model...")
            model.save(f"{config['save_path']}/{config['algorithm']}")

        print("-" * 60)

    total_time = (time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type = str, default = "config/medium_1.yaml")
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    train_env = SubprocVecEnv([make_env(cfg) for _ in range(cfg["train"]["num_train_envs"])])#my_config["num_train_envs"])])

    eval_env = DummyVecEnv([make_env(cfg)])

    # reward normalization
    train_env = VecNormalize(
        train_env,
        norm_obs=False,        # you can switch to True later, but RL on embeddings may not need it
        norm_reward=True,      # ★ the important part
        clip_reward=10.0       # prevents exploding gradients; tune as needed
    )

    # For eval, enable reward normalization but disable updating running stats
    eval_env = VecNormalize(
        eval_env,
        training=False,        # do not update running mean/std
        norm_obs=False,
        norm_reward=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

    if cfg["train"]["policy_network"] == "CnnPolicy":
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
    ALGOS = {
        "PPO": PPO
    }
    model = ALGOS[cfg["train"]["algorithm"]](
            cfg["train"]["policy_network"], 
            train_env, 
            verbose=1,
            device=device,
            tensorboard_log=cfg["tensorboard"]["run_id"],
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=cfg["train"]["n_steps"],
            batch_size=cfg["train"]["batch_size"],
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
    
    if cfg["wandb"]["enabled"]:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project = cfg["wandb"]["project"],
            name = cfg["wandb"]["run_id"],
            config = cfg["train"],
            sync_tensorboard=cfg["wandb"]["sync_tensorboard"],
            id = cfg["wandb"]["run_id"]
        )

    train(eval_env, model, cfg)
