import os
import cv2
import copy
import random
import tempfile
import numpy as np
from gymnasium import Env, spaces
from get_reward import evaluate_sequence_miou

class VideoEnv(Env):
    def __init__(self, video_dir="data/GOT10/train", frame_size=60, stack=3):
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

        # observation: (stacked channels, height, width)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.stack*3, self.frame_size, self.frame_size), # stack RGB 3 channels
            dtype=np.float32
        )

        self.frames = []            # name list of frames in the video
        self.video_length = 0       # number of frames in the video
        self.sliding_window = []    # sliding window of frames
        self.current_frame_idx = 0  # next frame index to read
        self.tmp_dir = None         # temporary directory for enhanced frames
        self.LQ_dir = None          # path to low quality video
        self.image = None           # current frame in full resolution

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # random select a video
        videos = [f for f in os.listdir(self.video_dir)]
        self.LQ_dir = os.path.join(self.video_dir, random.choice(videos), "degraded")
        self.frames = [f for f in os.listdir(self.LQ_dir) if f.endswith(".jpg")]
        self.frames = [os.path.join(self.LQ_dir, f) for f in self.frames]

        self.video_length = len(self.frames)
        if self.video_length < self.stack:
            raise RuntimeError("The video is shorter than the sliding window.")
        self.sliding_window = []
        self.current_frame_idx = 0
        self.close()  # clean up previous temp dir if any
        self.tmp_dir = tempfile.TemporaryDirectory()

        # read initial frames
        for idx in range(self.stack):
            self.image = cv2.imread(self.frames[self.current_frame_idx])
            self.current_frame_idx += 1
            # Save original frame to temporary directory
            if self.current_frame_idx < self.stack:
                path = os.path.join(self.tmp_dir.name, f"{self.current_frame_idx:08d}.jpg")
                cv2.imwrite(path, self.image)
            preprocessed_image = self._preprocess(self.image, resize=True)
            self.sliding_window.append(copy.deepcopy(preprocessed_image))

        return self._get_obs(), {}

    def step(self, action):
        # reward = 1.0  # testing
        done = False
        truncate = False
        info = {}
        
        # Apply enhancements to current frame based on action
        self.sliding_window[-1] = self._apply_enhancements(self.sliding_window[-1], action)

        # # Get the most recent frame
        # enhanced_frame = copy.deepcopy(self.sliding_window[-1])
        
        # # Apply each enhancement operation
        # enhanced_frame = self._apply_enhancements(enhanced_frame, action)
        
        # # Replace the last frame with enhanced version
        # self.sliding_window[-1] = copy.deepcopy(enhanced_frame)

        # Save enhanced frame to temporary directory
        path = os.path.join(self.tmp_dir.name, f"{self.current_frame_idx:08d}.jpg")
        enhanced_frame = self._preprocess(self.image, resize=False)
        enhanced_frame = self._apply_enhancements(enhanced_frame, action)
        enhanced_frame = self._postprocess(enhanced_frame)
        cv2.imwrite(path, enhanced_frame)
        
        # TODO: calculate reward
        miou_1 = evaluate_sequence_miou(self.tmp_dir, index=self.current_frame_idx)
        miou_0 = evaluate_sequence_miou(self.LQ_dir,  index=self.current_frame_idx)
        reward = miou_1 - miou_0
        
        info = {
            'action': action,
            'frame_index': self.current_frame_idx
        }
        
        # read next frame
        if self.current_frame_idx >= self.video_length:
            self.close()
            done = True # end of video
        else:
            self.sliding_window.pop(0)  
            self.image = cv2.imread(self.frames[self.current_frame_idx])
            self.current_frame_idx += 1
            preprocessed_image = self._preprocess(self.image, resize=True)
            self.sliding_window.append(copy.deepcopy(preprocessed_image))

        return self._get_obs(), reward, done, truncate, info

    def _preprocess(self, frame: np.ndarray, resize) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        if resize:                                      # resize to (frame_size, frame_size)
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
        frame = frame.astype(np.float32) / 255.0        # normalize to [0, 1]
        frame = np.transpose(frame, (2, 0, 1))          # (height, width, channels) to (channels, height, width)
        return frame
    
    def _postprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = np.transpose(frame, (1, 2, 0))
        frame = (frame * 255.0).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    
    def _get_obs(self):
        obs = np.concatenate(self.sliding_window, axis=0)
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

    # def _calculate_reward(self, enhanced_frame, original_frame):
    #     """
    #     Calculate reward based on improvement in tracking performance
        
    #     TODO: Implement this based on your single object tracking metric
    #     For now, returns a dummy reward
        
    #     You might want to:
    #     1. Run tracking on both enhanced and original frames
    #     2. Compare tracking confidence/accuracy
    #     3. Return the improvement as reward
    #     """
    #     # Placeholder reward calculation
    #     # You should replace this with actual tracking performance comparison
        
    #     # Example: negative of mean squared error (higher is better)
    #     # This is just a placeholder - replace with actual tracking metric
    #     reward = -np.mean((enhanced_frame - original_frame) ** 2)
        
    #     return float(reward)
    
    def close(self):
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
    
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
