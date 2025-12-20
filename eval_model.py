import os
import yaml
import torch
import argparse
import numpy as np
import csv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.env import VideoEnv

ALGOS = {"PPO": PPO, "A2C": A2C}

def make_env(cfg):
    def _init():
        return VideoEnv(
            data_dir=cfg["train"]["data_root"],
            val_dir=cfg["valid"]["data_root"],
            stack=cfg["train"]["stack"],
            action_repeat=cfg["train"]["action_repeat"],
            encoder=cfg["train"]["encoder"]
        )
    return _init

def run_evaluation(model_path, config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Replicate environment setup from train.py
    eval_env = DummyVecEnv([make_env(cfg)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=True)

    # Load model
    model = ALGOS[cfg["train"]["algorithm"]].load(model_path, env=eval_env, device=device)

    # Logging setup
    action_log_cfg = cfg.get("action_logging", {})
    log_actions = action_log_cfg.get("enabled", False) and action_log_cfg.get("log_eval", False)
    log_path = action_log_cfg.get("eval_log_path", "logs/actions/actions_val.csv")
    
    if log_actions:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "episode", "step", "env_id", "equalize", "brightness", "contrast", "sharpen", "gamma"])

    eval_episode_num = cfg["train"]["eval_episode_num"]
    visualize_index = cfg["visualize"]["index"]
    total_reward = 0.0

    for seed in range(eval_episode_num):
        done = False
        is_vis = seed in visualize_index
        
        # Consistent with train.py evaluation logic
        eval_env.seed(seed)
        reset_options = {"eval_id": seed, "vis_flag": is_vis, "epoch": "eval_run"}
        
        # VecEnv.reset() with options
        obs = eval_env.env_method("reset", options=reset_options)[0][0]
        
        ep_reward = 0.0
        step_idx = 0

        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Ensure action is formatted correctly for the step and logging
            if action.ndim == 0: 
                action = np.array([action])
            
            # Log actions
            if log_actions:
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    # For a single env, action is typically (5,)
                    writer.writerow(["eval", seed, step_idx, 0, *action.flatten()])
            
            # Step the environment
            obs, reward, done, info = eval_env.step(action if action.ndim > 1 else [action])
            
            ep_reward += reward[0]
            step_idx += 1

        print(f"Episode {seed} Finished | Reward: {ep_reward:.4f}")
        total_reward += ep_reward

    print(f"\nFinal Average Reward: {total_reward / eval_episode_num:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config/medium_PPO_ResNet_eval.yaml")
    args = parser.parse_args()
    run_evaluation(args.model_path, args.config_path)