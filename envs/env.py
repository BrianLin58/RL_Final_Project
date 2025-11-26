import os
import cv2
import random
import numpy as np
from gymnasium import Env, spaces

class VideoEnv(Env):
    def __init__(self, video_dir="D:/Courses/114-1/RL/Final_Project/videos", frame_size=64, stack=3):
        self.video_dir = video_dir    # path to video directory
        self.frame_size = frame_size  # height and width of each frame
        self.stack = stack            # sliding window size

        # continuous action: modify shape(x,) to indicate action dimension
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

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

    def close(self):
        if self.cap is not None:
            self.cap.release()
    
    def brightness(self, value):
        pass  # TODO: implement brightness adjustment

    def contrast(self, value):
        pass  # TODO: implement contrast adjustment

    def gamma(self, value):
        pass  # TODO: implement gamma correction
