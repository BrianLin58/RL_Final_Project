import os
import cv2
import random
import numpy as np
from gymnasium import Env, spaces

class VideoEnv(Env):
    def __init__(self, video_dir="videos", frame_size=64, stack=3):
        self.video_dir = video_dir    # path to video directory
        self.frame_size = frame_size  # height and width of each frame
        self.stack = stack            # sliding window size

        # continuous action: modify shape(x,) to indicate action dimension
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([
            2,  # Equalize: 0=not apply, 1=apply CLAHE
            4,  # Brightness: {-0.05, 0, +0.05, +0.1}
            4,  # Contrast: {0.9, 1.0, 1.1, 1.2}
            3,  # Sharpen: {0, 0.5, 1.0}
            3   # Gamma: {0.8, 1.0, 1.2}
        ])
        
        self.brightness_values = [-0.05, 0, 0.05, 0.1]
        self.contrast_values = [0.9, 1.0, 1.1, 1.2]
        self.sharpen_values = [0, 0.5, 1.0]
        self.gamma_values = [0.8, 1.0, 1.2]

        # observation: (stack, height, width)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.stack*3, self.frame_size, self.frame_size), # stack RGB 3 channels
            dtype=np.float32
        )

        self.cap = None     # video capture object
        self.frames = []    # sliding window of frames

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # random select a video
        videos = [f for f in os.listdir(self.video_dir) if f.endswith(".mp4")]
        video = os.path.join(self.video_dir, random.choice(videos))

        self.close()
        self.cap = cv2.VideoCapture(video)
        self.frames = []

        # read initial frames
        for frame_idx in range(self.stack):
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("The video is shorter than the sliding window.")
            self.frames.append(self._preprocess(frame))

        return self._get_obs(), {}

    def step(self, action):
        # TODO: modify current frame according to action

        # TODO: calculate reward
        reward = 1.0  # testing
        done = False
        truncate = False
        info = {}
        
        # read next frame
        ok, frame = self.cap.read()
        if ok:
            self.frames.pop(0)
            self.frames.append(self._preprocess(frame))
        else:
            done = True # end of video

        return self._get_obs(), reward, done, truncate, info

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                # convert BGR to RGB
        frame = cv2.resize(frame, (self.frame_size, self.frame_size)) # resize to (frame_size, frame_size)
        frame = frame.astype(np.float32) / 255.0                      # normalize to [0, 1]
        frame = np.transpose(frame, (2, 0, 1))                        # (height, width, channels) to (channels, height, width)
        return frame

    def _get_obs(self):
        obs = np.concatenate(self.frames, axis=0)
        return obs.astype(np.float32)
    
    def _apply_enhancements(self, frame, action):
        """
        Apply enhancement operations based on discrete action values
        frame: (C, H, W) in range [0, 1]
        action: [equalize, brightness, contrast, sharpen, gamma]
        """
        # Convert from (C, H, W) to (H, W, C) for processing
        frame = np.transpose(frame, (1, 2, 0))
        
        # Convert to uint8 for some operations
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # 1. Equalize (CLAHE)
        if action[0] == 1:
            frame_uint8 = self._apply_clahe(frame_uint8)
        
        # Convert back to float for other operations
        frame = frame_uint8.astype(np.float32) / 255.0
        
        # 2. Brightness
        brightness_delta = self.brightness_values[action[1]]
        frame = self._apply_brightness(frame, brightness_delta)
        
        # 3. Contrast
        contrast_alpha = self.contrast_values[action[2]]
        frame = self._apply_contrast(frame, contrast_alpha)
        
        # 4. Sharpen
        sharpen_lambda = self.sharpen_values[action[3]]
        frame = self._apply_sharpen(frame, sharpen_lambda)
        
        # 5. Gamma Correction
        gamma_value = self.gamma_values[action[4]]
        frame = self._apply_gamma(frame, gamma_value)
        
        # Clip values to [0, 1]
        frame = np.clip(frame, 0.0, 1.0)
        
        # Convert back to (C, H, W)
        frame = np.transpose(frame, (2, 0, 1))
        
        return frame.astype(np.float32)
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
    
    def _apply_clahe(self, frame):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Apply CLAHE to each channel
        for i in range(3):
            frame[:, :, i] = clahe.apply(frame[:, :, i])
        
        return frame

    def _apply_brightness(self, frame, delta):
        """Apply brightness adjustment: I' = I + delta"""
        return frame + delta

    def _apply_contrast(self, frame, alpha):
        """Apply contrast adjustment: I' = alpha * (I - 0.5) + 0.5"""
        return alpha * (frame - 0.5) + 0.5

    def _apply_sharpen(self, frame, lambda_val):
        """Apply sharpening: I' = I + lambda * (I - blur(I))"""
        if lambda_val == 0:
            return frame
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
        
        # Sharpen
        return frame + lambda_val * (frame - blurred)

    def _apply_gamma(self, frame, gamma):
        """Apply gamma correction: I' = I^gamma"""
        return np.power(frame, gamma)
