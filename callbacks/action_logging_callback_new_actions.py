import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ActionLoggingCallback(BaseCallback):
    """
    Logs actions taken by the agent.
    Each row:
      epoch, episode, step, env_id, a0, a1, a2, a3, a4, a5
    """

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path

        self.episode_counts = None
        self.step_counts = None
        self.epoch = 0

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        self.episode_counts = np.zeros(self.training_env.num_envs, dtype=int)
        self.step_counts = np.zeros(self.training_env.num_envs, dtype=int)

        # Create CSV + header
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "episode",
                "step",
                "env_id",
                "denoise",
                "deblock",
                "sharpen",
                "gamma",
                "CLAHE",
            ])

    def set_epoch(self, epoch: int):
        """Call this manually from the training loop."""
        self.epoch = epoch

    def _on_step(self) -> bool:
        actions = self.locals["actions"]      # shape: (n_envs, 5)
        dones = self.locals["dones"]

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)

            for env_id in range(len(actions)):
                action = actions[env_id]

                writer.writerow([
                    self.epoch,
                    self.episode_counts[env_id],
                    self.step_counts[env_id],
                    env_id,
                    int(action[0]),
                    int(action[1]),
                    int(action[2]),
                    int(action[3]),
                    int(action[4]),
                ])

                self.step_counts[env_id] += 1

                if dones[env_id]:
                    self.episode_counts[env_id] += 1
                    self.step_counts[env_id] = 0

        return True
