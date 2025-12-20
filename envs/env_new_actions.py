import os
import cv2
import copy
import torch
import random
import time
import numpy as np
from gymnasium import Env, spaces
# Ensure these imports match your actual file structure
from get_reward import evaluate_sequence_miou, evaluate_sequence_miou_from_frames
from utils.parse_bbox_files import parse_bbox_from_files
from encoder.placeholder_encoder import PlaceholderEncoder, ResNet18Encoder
from encoder.dbcnn_feature_wrapper import DBCNNEncoder
from calc_similarity import calc_miou_from_boxes


class VideoEnv(Env):
    def __init__(self, data_dir="data/GOT10/train", val_dir="data/GOT10/val", frame_size=60, stack=3, action_repeat=32, encoder=None):
        self.data_dir = data_dir      # path to training data directory
        self.val_dir = val_dir        # path to validation data directory
        self.frame_size = frame_size  # height and width of each frame
        self.stack = stack            # sliding window size
        self.action_repeat = action_repeat  # number of frames to repeat each action
        self.vis_flag = False
        
        self.current_action = None
        self.action_counter = 0

        # --- NEW ACTION SPACE ALIGNED WITH DEGRADATION ---
        # 0: Denoise (0=None, 1=Gaussian, 2=Median) -> Counters GaussNoise
        # 1: Deblock (0=None, 1=Bilateral) -> Counters JPEG Artifacts
        # 2: Sharpen (0=None, 1=Weak, 2=Medium, 3=Strong) -> Counters Blur
        # 3: Tone/Gamma (0=Darker, 1=Normal, 2=Brighter) -> Counters Visibility
        # 4: Contrast (0=None, 1=CLAHE) -> Counters Downscale/Flatness
        self.action_space = spaces.MultiDiscrete([3, 2, 4, 3, 2])
        
        # Mappings for the discrete actions
        self.sharpen_values = [0.0, 0.5, 1.0, 1.5]
        self.gamma_values   = [0.8, 1.0, 1.2]

        self.sample_dir = None         
        self.lq_dir = None             
        self.lq_frame_paths = []       
        self.gt_boxes = []             
        self.perturbed_boxes = []      
        self.lq_boxes = []             
        self.video_length = 0          
        self.frame_index = 0           
        self.all_lq_frames = []        
        self.all_perturbed_frames = [] 
        self.episode_start_time = 0

        # Encoder Setup
        if encoder == "ResNet18":
            self.encoder = ResNet18Encoder()
            self.feature_dim = 512
        elif encoder == "DBCNN":
            self.encoder = DBCNNEncoder(use_x1=False, use_x2=True)
            self.feature_dim = 128
        else:
            self.encoder = PlaceholderEncoder()
            self.feature_dim = 512

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
            
        # Observation Space: (stack, feature_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.stack, self.feature_dim),
            dtype=np.float32
        )

        try:
            self.tracker_perturbed = cv2.TrackerCSRT_create()
            self.tracker_lq = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker_perturbed = cv2.legacy.TrackerCSRT_create()
            self.tracker_lq = cv2.legacy.TrackerCSRT_create()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_start_time = time.time()
        reset_start_time = time.time()

        # Select Video
        if options is None: # training mode
            video_list = [f for f in os.listdir(self.data_dir)]
            while True:
                self.sample_dir = os.path.join(self.data_dir, random.choice(video_list))
                # Basic check for existence
                if os.path.exists(os.path.join(self.sample_dir, "groundtruth.txt")):
                    break
        else: # evaluation mode
            eval_id = options.get("eval_id", 0)
            video_list = sorted([f for f in os.listdir(self.val_dir)])
            if not video_list:
                raise RuntimeError(f"No videos found in {self.val_dir}")
            
            video_name = video_list[eval_id % len(video_list)]
            self.sample_dir = os.path.join(self.val_dir, video_name)

            self.vis_flag = options.get("vis_flag", False)
            print(f"[DEBUG] self.vis_flag is {self.vis_flag}")
            if self.vis_flag:
                print(f"[DEBUG] Will visualize {video_name}")
                epoch = options.get("epoch", 0)
                self.visualize_dir = f"visualization_DBCNN_01/{video_name}/ep{epoch}"
                os.makedirs(os.path.join(self.visualize_dir, "lq"), exist_ok=True)
                os.makedirs(os.path.join(self.visualize_dir, "perturbed"), exist_ok=True)

        # Load Data
        self.lq_dir = os.path.join(self.sample_dir, "degraded")
        if not os.path.exists(self.lq_dir):
             raise RuntimeError(f"Degraded folder not found: {self.lq_dir}")

        self.lq_frame_paths = sorted([os.path.join(self.lq_dir, f) for f in os.listdir(self.lq_dir) if f.endswith(".jpg")])
        self.gt_boxes, _ = parse_bbox_from_files(os.path.join(self.sample_dir, "groundtruth.txt"))

        self.video_length = len(self.lq_frame_paths)
        if self.video_length < self.stack:
            raise RuntimeError("The video is shorter than the sliding window.")

        # Read all frames into memory
        self.all_lq_frames = [cv2.imread(f) for f in self.lq_frame_paths]
        
        # Reset Buffers
        self.all_perturbed_frames = [] 
        self.lq_boxes = []
        self.perturbed_boxes = []

        # Initialize stack with first frames (unperturbed for history)
        # We need stack-1 frames of history before the first "active" frame
        for idx in range(self.stack - 1): 
            self.all_perturbed_frames.append(copy.deepcopy(self.all_lq_frames[idx]))
        
        self.frame_index = self.stack - 1 

        # Initialize Tracker
        try:
            # We track on the *perturbed* history, which is just LQ initially
            self.tracker_perturbed.init(self.all_perturbed_frames[0], self.gt_boxes[0])
        except cv2.error:
            print("[WARNING] Tracker init failed. Resetting...")
            return self.reset()

        self.perturbed_boxes.append(self.gt_boxes[0]) 

        # Update tracker through the initial stack history
        for idx in range(1, self.stack - 1):
            success, bbox = self.tracker_perturbed.update(self.all_perturbed_frames[idx]) 
            self.perturbed_boxes.append(bbox if success else self.perturbed_boxes[-1])

        reset_end_time = time.time()
        print(f"[DEBUG] Reset time: {reset_end_time - reset_start_time:.4f}s")
        
        return self._get_obs(), {}


    def step(self, action):
        done = False
        truncate = False
        info = {}
        step_start_time = time.time()
        
        self.current_action = action
        
        # --- BATCH PROCESSING LOGIC ---
        # 1. Identify the chunk of frames to process
        start_idx = self.frame_index
        end_idx = min(self.frame_index + self.action_repeat, self.video_length)
        
        # 2. Get frames (uint8 BGR)
        raw_frames_chunk = np.array(self.all_lq_frames[start_idx : end_idx], dtype=np.uint8)
        
        # 3. Apply enhancements (Batch)
        enhanced_chunk = self._apply_enhancements(raw_frames_chunk, self.current_action)
        
        # 4. Update Tracker & Store Frames
        for k in range(enhanced_chunk.shape[0]):
            enhanced_frame = enhanced_chunk[k]
            
            # Update Tracker
            success, bbox = self.tracker_perturbed.update(enhanced_frame)
            if success:
                self.perturbed_boxes.append(bbox)
            else:
                self.perturbed_boxes.append(self.perturbed_boxes[-1])
                # print(f"[WARNING] Tracking failed at frame {self.frame_index + 1}")
            
            # Store Result
            self.all_perturbed_frames.append(enhanced_frame)
            self.frame_index += 1    

        print(f"[DEBUG] Step applied. Sample: {self.sample_dir} | Frames: {start_idx}-{end_idx}")
        
        # --- EPISODE END LOGIC ---
        if self.frame_index >= self.video_length:
            print(f"[DEBUG] Episode finished. Calculating Reward...")

            # 1. Track original LQ frames for baseline
            self.tracker_lq.init(self.all_lq_frames[0], self.gt_boxes[0])
            self.lq_boxes.append(self.gt_boxes[0])
            
            for idx in range(1, self.video_length):
                success, bbox = self.tracker_lq.update(self.all_lq_frames[idx])
                self.lq_boxes.append(bbox if success else self.lq_boxes[-1])

            # 2. Calculate mIoU Improvement
            # Note: calc_miou_from_boxes expects lists of (x,y,w,h)
            miou_before = calc_miou_from_boxes(self.gt_boxes, self.lq_boxes, self.video_length)
            miou_after = calc_miou_from_boxes(self.gt_boxes, self.perturbed_boxes, self.video_length)
            
            reward = miou_after - miou_before
            
            # 3. Visualization (Optional)
            if self.vis_flag:
                for i in range(self.video_length):
                    # Save Perturbed
                    cv2.imwrite(os.path.join(self.visualize_dir, 'perturbed', f"{(i+1):08d}.jpg"), 
                                self.all_perturbed_frames[i])
                    # Save Original LQ (Use PNG to avoid double compression if needed, but JPG matches request)
                    cv2.imwrite(os.path.join(self.visualize_dir, 'lq', f"{(i+1):08d}.jpg"), 
                                self.all_lq_frames[i])

            done = True
            print(f"[INFO] Reward: {reward:.4f} (mIoU: {miou_before:.4f} -> {miou_after:.4f})")
            print(f"[INFO] Episode Time: {time.time() - self.episode_start_time:.2f}s")

        else:
            # Intermediate step reward is 0 (sparse reward setting)
            reward = 0.0

        info = {
            'action': action,
            'next_frame_index': self.frame_index
        }
        
        return self._get_obs(), reward, done, truncate, info

    def _preprocess(self, frame: np.ndarray, resize) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        if resize:                                      
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
        frame = frame.astype(np.float32) / 255.0        
        frame = np.transpose(frame, (2, 0, 1))          
        return frame
    
    def _get_obs(self):
        # Retrieve the last 'stack' frames from the perturbed history
        # If we are at the very end, just take the last available ones
        if len(self.all_perturbed_frames) < self.stack:
             # Fallback if somehow history is short (shouldn't happen with correct reset)
             state = self.all_perturbed_frames
        else:
             state = self.all_perturbed_frames[-self.stack:]

        processed = []
        ENC_SIZE = 224   

        for frame_bgr in state:
            # Resize and Normalize for Encoder
            chw = self._preprocess(frame_bgr, resize=False)
            hwc = np.transpose(chw, (1, 2, 0))
            hwc_small = cv2.resize(hwc, (ENC_SIZE, ENC_SIZE), interpolation=cv2.INTER_LINEAR)
            chw_small = np.transpose(hwc_small, (2, 0, 1))
            processed.append(chw_small.astype(np.float32))

        # Check if we have enough frames to stack
        if len(processed) < self.stack:
            # Pad with the first frame if needed
            padding = [processed[0]] * (self.stack - len(processed))
            processed = padding + processed

        frames = np.stack(processed, axis=0)  # (stack, 3, 224, 224)

        with torch.no_grad():
            x = torch.from_numpy(frames).to(self.device)
            obs = self.encoder(x).detach().cpu().numpy()

        return obs.astype(np.float32)

    
    def _apply_enhancements(self, batch_frames, action):
        """
        Processes a batch of frames based on the 5 discrete actions.
        batch_frames: (N, H, W, 3) uint8 BGR
        action: [Denoise, Deblock, Sharpen, Tone, Contrast]
        """
        # Unpack Action Indices
        denoise_idx = action[0] # 0=None, 1=Gaussian, 2=Median
        deblock_idx = action[1] # 0=None, 1=Bilateral
        sharpen_idx = action[2] # 0=0, 1=0.5, 2=1.0, 3=1.5
        gamma_idx   = action[3] # 0=0.8, 1=1.0, 2=1.2
        clahe_idx   = action[4] # 0=None, 1=CLAHE

        # Get Float Values
        sharpen_val = self.sharpen_values[sharpen_idx]
        gamma_val   = self.gamma_values[gamma_idx]

        processed_batch = []

        for frame_bgr in batch_frames:
            # --- 1. Operations in uint8 domain (Structure & Contrast) ---
            
            # A. Denoise (Target: GaussNoise) - DO THIS FIRST
            if denoise_idx == 1:
                # Gaussian Blur (Fast, good for grain)
                frame_bgr = cv2.GaussianBlur(frame_bgr, (3, 3), 0)
            elif denoise_idx == 2:
                # Median Blur (Best for Salt & Pepper / Sparkle noise)
                frame_bgr = cv2.medianBlur(frame_bgr, 3)

            # B. Deblock (Target: JPEG Artifacts)
            if deblock_idx == 1:
                # Bilateral Filter: Smooths flat areas (blocks) but keeps edges
                # d=5, sigmaColor=75, sigmaSpace=75 are standard opencv values
                frame_bgr = cv2.bilateralFilter(frame_bgr, 5, 75, 75)

            # C. Contrast / CLAHE (Target: Downscale flatness)
            if clahe_idx == 1:
                # Convert to LAB, apply to L channel
                lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                frame_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # --- 2. Operations in float32 domain (Math) ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # D. Tone Mapping (Gamma) (Target: General Visibility)
            if gamma_val != 1.0:
                # Safe power operation
                frame_rgb = np.power(frame_rgb + 1e-6, gamma_val)

            # E. Sharpening (Target: Blur) - DO THIS LAST
            if sharpen_val > 0.0:
                # Unsharp Masking logic
                blurred_s = cv2.GaussianBlur(frame_rgb, (0, 0), sigmaX=1.0, sigmaY=1.0)
                frame_rgb = frame_rgb + sharpen_val * (frame_rgb - blurred_s)

            # --- 3. Finalize ---
            frame_rgb = np.clip(frame_rgb, 0.0, 1.0)
            final_bgr = (frame_rgb * 255).astype(np.uint8)
            final_bgr = cv2.cvtColor(final_bgr, cv2.COLOR_RGB2BGR)
            
            processed_batch.append(final_bgr)

        return np.array(processed_batch)

if __name__ == "__main__":
    # Simple Test Block
    env = VideoEnv(encoder="Placeholder") # Use Placeholder to avoid loading heavy models
    print("Action Space:", env.action_space)
    
    # Fake Reset
    # Note: This will fail if directories don't exist, which is expected in a standalone test
    try:
        obs, _ = env.reset()
        print("Observation Shape:", obs.shape)
        
        # Fake Step
        action = env.action_space.sample()
        print("Sample Action:", action)
        obs, reward, done, _, info = env.step(action)
        print("Step Reward:", reward)
    except Exception as e:
        print(f"Test skipped due to missing data: {e}")