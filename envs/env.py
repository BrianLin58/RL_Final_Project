import os
import cv2
import copy
import torch
import random
import tempfile
import numpy as np
from gymnasium import Env, spaces
from get_reward import evaluate_sequence_miou, evaluate_sequence_miou_from_frames
from utils.parse_bbox_files import parse_bbox_from_files
from encoder.placeholder_encoder import PlaceholderEncoder, ResNet18Encoder
from calc_similarity import calc_miou_from_boxes

class VideoEnv(Env):
    def __init__(self, data_dir="data/GOT10/train", val_dir = "data/GOT10/val", frame_size=60, stack=3):
        self.data_dir = data_dir      # path to training data directory
        self.val_dir = val_dir        # path to val data directory
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
            low=-np.inf, high=np.inf,
            shape=(self.stack, 512),
            # low=0.0, high=1.0,
            # shape=(self.stack * 3, self.frame_size, self.frame_size), # stack RGB 3 channels
            # shape = (self.stack * 3, 1080, 1920), # impossible :)
            dtype=np.float32
        )

        self.sample_dir = None         # path to current video sample
        self.lq_dir = None             # path to low quality video
        self.lq_frame_paths = []       # list of paths to low quality jpg
        self.gt_boxes = []             # groundtruth bounding boxes
        self.perturbed_boxes = []      # predicted bbox for perturbed frames
        self.lq_boxes = []             # predicted bbox for original lq frames
        self.video_length = 0          # number of frames in the video
        self.sliding_window = []       # sliding window of frames, each (C, H, W) float32 [0,1]
        self.frame_index = 0           # next frame index to read
        # self.tmp_dir = None            # temporary directory for enhanced frames
        self.image = None              # current frame in full resolution
        self.all_lq_frames = []        # stores BGR, HWC data
        self.all_perturbed_frames = [] # stores BGR, HWC data

        # self.encoder = PlaceholderEncoder(out_size = self.frame_size)
        self.encoder = ResNet18Encoder()

        self.tracker_perturbed = cv2.TrackerCSRT_create() # use a global tracker for perturbed frames
        self.tracker_lq = cv2.TrackerCSRT_create() # use another global tracker for lq frames

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # random select a video
        if options is None:
            video_list = [f for f in os.listdir(self.data_dir)]
            self.sample_dir = os.path.join(self.data_dir, random.choice(video_list))
        elif options:
            eval_id = options.get("eval_id", None)
            video_list = [f for f in os.listdir(self.val_dir)]
            video_list.sort()
            self.sample_dir = os.path.join(self.val_dir, video_list[eval_id % len(video_list)])
        self.lq_dir = os.path.join(self.sample_dir, "degraded")
        self.lq_frame_paths = [f for f in os.listdir(self.lq_dir) if f.endswith(".jpg")]
        self.lq_frame_paths = [os.path.join(self.lq_dir, f) for f in self.lq_frame_paths] # self.lq_frame_paths is a list of paths to low quality jpg
        self.gt_boxes, _ = parse_bbox_from_files(os.path.join(self.sample_dir, "groundtruth.txt"))
        # print(f"[DEBUG] Getting groundtruth bbox...")

        self.video_length = len(self.lq_frame_paths)
        if self.video_length < self.stack:
            raise RuntimeError("The video is shorter than the sliding window.")
        self.sliding_window = []
        self.frame_index = 0
        # self.close()  # clean up previous temp dir if any
        # self.tmp_dir = tempfile.TemporaryDirectory()
        # print(f"[DEBUG] Created temporary directory at {self.tmp_dir.name}")

        self.all_lq_frames = [] # stores BGR, HWC data
        for frame_path in self.lq_frame_paths:
            self.all_lq_frames.append(cv2.imread(frame_path))

        self.all_perturbed_frames = [] # stores BGR, HWC data

        self.lq_boxes = []
        self.perturbed_boxes = []

        # read initial frames
        for idx in range(self.stack):
            self.image = self.all_lq_frames[self.frame_index]
            self.frame_index += 1
            if idx < self.stack:
                self.all_perturbed_frames.append(copy.deepcopy(self.image))
            self.sliding_window.append(self._preprocess(self.image, resize = False))

        # initialize tracker
        self.tracker_perturbed.init(self.all_perturbed_frames[0], self.gt_boxes[0])
        self.perturbed_boxes.append(self.gt_boxes[0]) # pred_box[0] = gt_box[0]
        print(f"[DEBUG] Tracking {self.sample_dir}'s initial {self.stack} frames...")
        for idx in range(1, self.stack):
            success, bbox = self.tracker_perturbed.update(self.all_perturbed_frames[idx]) # also append predicted boxes of initial frames
            if success:
                self.perturbed_boxes.append(bbox)
            else:
                self.perturbed_boxes.append(self.perturbed_boxes[-1])
                print(f"[WARNING] Failed to track the {idx}th lq frame. Using previous value.")

        # finish all lq boxes predict upon init
        self.tracker_lq.init(self.all_lq_frames[0], self.gt_boxes[0])
        self.lq_boxes.append(self.gt_boxes[0])
        print(f"[DEBUG] Tracking ALL lq frames for {self.sample_dir}...")
        for idx in range(1, self.video_length):
            success, bbox = self.tracker_lq.update(self.all_lq_frames[idx])
            if success:
                self.lq_boxes.append(bbox)
            else:
                self.lq_boxes.append(self.lq_boxes[-1])
                print(f"[WARNING] Failed to track The {idx}th lq frame. Using previous value.")
        print(f"[DEBUG] Finished tracking all lq frames for {self.sample_dir}")

        return self._get_obs(), {}

        # read initial frames
        # for idx in range(self.stack):
        #     self.image = cv2.imread(self.lq_frame_paths[self.frame_index])
        #     self.frame_index += 1
        #     # Save original frame to temporary directory
        #     if self.frame_index < self.stack:
        #         path = os.path.join(self.tmp_dir.name, f"{self.frame_index:08d}.jpg")
        #         cv2.imwrite(path, self.image)
        #     preprocessed_image = self._preprocess(self.image, resize=True)
        #     self.sliding_window.append(copy.deepcopy(preprocessed_image))

        # return self._get_obs(), {}

    def step(self, action):
        # reward = 1.0  # testing
        done = False
        truncate = False
        info = {}
        
        # Apply enhancements to current frame based on action
        self.sliding_window[-1] = self._apply_enhancements(self.sliding_window[-1], action) # RGB, CHW

        # Save enhanced frame to temporary directory
        # path = os.path.join(self.tmp_dir.name, f"{self.frame_index:08d}.jpg")
        # enhanced_frame = self._preprocess(self.image, resize=False)
        # enhanced_frame = self._apply_enhancements(enhanced_frame, action) # enhanced_frame is now in shape (C, H, W)
        # enhanced_frame = self._postprocess(enhanced_frame) # enhanced_frame is now BGR, shape (H, W, C)
        # cv2.imwrite(path, enhanced_frame) # TODO: write a flag to do this

        self.all_perturbed_frames.append(self._postprocess(self.sliding_window[-1]))
        # TODO: write a flag to store frame here

        
        # TODO: calculate reward
        # tmp_parent = os.path.dirname(self.tmp_dir.name)
        # lq_parent = os.path.dirname(self.lq_dir)
        print(f"self.sample_dir = {self.sample_dir}")
        # miou_1 = evaluate_sequence_miou(self.tmp_dir.name, os.path.join(self.sample_dir, "groundtruth.txt"), index=self.frame_index)
        # miou_0 = evaluate_sequence_miou(self.lq_dir, os.path.join(self.sample_dir, "groundtruth.txt"), index=self.frame_index)
        # miou_after  = evaluate_sequence_miou_from_frames(self.all_perturbed_frames, self.gt_boxes, index = self.frame_index)
        # miou_before = evaluate_sequence_miou_from_frames(self.all_lq_frames, self.gt_boxes, index = self.frame_index)
        # reward = miou_after - miou_before

        print(f"[DEBUG] Calculating reward for the {self.frame_index}th frame.") #3
        # print(f"[DEBUG] Below shows the lq_boxes")
        # print(self.lq_boxes)
        miou_before = calc_miou_from_boxes(self.gt_boxes, self.lq_boxes, self.frame_index)

        success, bbox = self.tracker_perturbed.update(self.all_perturbed_frames[-1]) # also append predicted boxes of initial frames
        if success:
            self.perturbed_boxes.append(bbox)
        else:
            self.perturbed_boxes.append(self.perturbed_boxes[-1])
            print(f"[WARNING] Failed to track the {self.frame_index}th perturbed frame. Using previous value.")
        miou_after = calc_miou_from_boxes(self.gt_boxes, self.perturbed_boxes, self.frame_index)
        reward = miou_after - miou_before
        
        print(f"reward = {reward}")
        
        info = {
            'action': action,
            'frame_index': self.frame_index
        }
        
        # read next frame
        if self.frame_index >= self.video_length:
            # self.close()
            done = True # end of video
        # else:
        #     self.sliding_window.pop(0)  
        #     self.image = cv2.imread(self.lq_frame_paths[self.frame_index])
        #     self.frame_index += 1
        #     preprocessed_image = self._preprocess(self.image, resize=True)
        #     self.sliding_window.append(copy.deepcopy(preprocessed_image))
        else:
            self.sliding_window.pop(0)
            self.image = self.all_lq_frames[self.frame_index]
            self.frame_index += 1 # too disgusting # I'm sorry :(
            self.sliding_window.append(self._preprocess(self.image, resize = False))

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
        # obs = np.concatenate(self.sliding_window, axis=0)
        # obs = self.encoder.encode(self.sliding_window)
        with torch.no_grad():
            obs = self.encoder(torch.tensor(self.sliding_window))  # (stack, feature_dim)
            obs = obs.detach().cpu().numpy()
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
    
    # def close(self):
    #     if self.tmp_dir is not None:
    #         self.tmp_dir.cleanup()
    
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

    def visualize(self, seed):
        visualize_dir = f"visualization/{seed}/"
        os.makedirs(os.path.join(visualize_dir, "lq"), exist_ok=True)
        os.makedirs(os.path.join(visualize_dir, "perturbed"), exist_ok=True)

        for frame_id, frame in enumerate(self.all_lq_frames):
            visualize_path = os.path.join(visualize_dir, "lq", f"{(frame_id + 1):08d}.jpg")
            cv2.imwrite(visualize_path, frame)
        print(f"[INFO] Done visualization of lq frames index {seed} in directory {visualize_dir}")

        for frame_id, frame in enumerate(self.all_perturbed_frames):
            visualize_path = os.path.join(visualize_dir, "perturbed", f"{(frame_id + 1):08d}.jpg")
            cv2.imwrite(visualize_path, frame)
        print(f"[INFO] Done visualization of perturbed frames {seed} in directory {visualize_dir}")

