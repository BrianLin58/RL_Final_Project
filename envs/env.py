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
        self.original_frames = []
        self.current_frame_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # random select a video
        videos = [f for f in os.listdir(self.video_dir) if f.endswith(".mp4")]
        video = os.path.join(self.video_dir, random.choice(videos))

        self.close()
        self.cap = cv2.VideoCapture(video)
        self.frames = []
        self.original_frames = []    # original frames for reward calculation
        self.current_frame_idx = 0

        # read initial frames
        for frame_idx in range(self.stack):
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("The video is shorter than the sliding window.")
            original = self._preprocess(frame)
            self.original_frames.append(original.copy())
            self.frames.append(original)

        return self._get_obs(), {}

    def step(self, action):
        # TODO: modify current frame according to action

        # TODO: calculate reward
        reward = 1.0  # testing
        done = False
        truncate = False
        info = {}
        
        # Apply enhancements to current frame based on action
        # Get the most recent frame
        enhanced_frame = self.frames[-1].copy()
        
        # Apply each enhancement operation
        enhanced_frame = self._apply_enhancements(enhanced_frame, action)
        
        # Replace the last frame with enhanced version
        self.frames[-1] = enhanced_frame
        
        reward  = self._calculate_reward(enhanced_frame, self.original_frames[-1])
        
        info = {
            'action': action,
            'frame_index': self.current_frame_idx
        }
        
        # read next frame
        ok, frame = self.cap.read()
        if ok:
            self.current_frame_idx += 1
            original = self._preprocess(frame)
            self.frames.pop(0)
            self.original_frames.pop(0)
            
            self.frames.append(original.copy())
            self.original_frames.append(original)
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

    def _calculate_reward(self, enhanced_frame, original_frame):
        """
        Calculate reward based on improvement in tracking performance
        
        TODO: Implement this based on your single object tracking metric
        For now, returns a dummy reward
        
        You might want to:
        1. Run tracking on both enhanced and original frames
        2. Compare tracking confidence/accuracy
        3. Return the improvement as reward
        """
        # Placeholder reward calculation
        # You should replace this with actual tracking performance comparison
        
        # Example: negative of mean squared error (higher is better)
        # This is just a placeholder - replace with actual tracking metric
        reward = -np.mean((enhanced_frame - original_frame) ** 2)
        
        return float(reward)
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
    
    def _apply_clahe(self, frame):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Create a copy to avoid modifying original
        result = np.zeros_like(frame)
        
        # Apply CLAHE to each channel
        for i in range(3):
            result[:, :, i] = clahe.apply(frame[:, :, i])
        
        return result

    def _apply_brightness(self, frame, delta):
        """Apply brightness adjustment: I' = I + delta"""
        result = frame + delta
        return np.clip(result, 0.0, 1.0)

    def _apply_contrast(self, frame, alpha):
        """Apply contrast adjustment: I' = alpha * (I - 0.5) + 0.5"""
        result = alpha * (frame - 0.5) + 0.5
        return np.clip(result, 0.0, 1.0)

    def _apply_sharpen(self, frame, lambda_val):
        """Apply sharpening: I' = I + lambda * (I - blur(I))"""
        if lambda_val == 0:
            return frame
        
        # Apply Gaussian blur with error handling
        try:
            blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
        except:
            # If blur fails, return original
            return frame
        
        # Sharpen
        result = frame + lambda_val * (frame - blurred)
        
        # Remove any NaN and clip
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(result, 0.0, 1.0)

    def _apply_gamma(self, frame, gamma):
        """Apply gamma correction: I' = I^gamma"""
        # Add small epsilon to avoid issues with zero values
        epsilon = 1e-7
        frame_safe = np.clip(frame, epsilon, 1.0)
        result = np.power(frame_safe, gamma)
        
        # Remove any NaN that might appear
        result = np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
        return np.clip(result, 0.0, 1.0)
