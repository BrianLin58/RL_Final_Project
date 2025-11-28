import cv2
import numpy as np

class PlaceholderEncoder:
    def __init__(self, out_size=60):
        self.out_size = out_size

    def encode(self, sliding_window):
        """
        sliding_window: list of frames, each (C, H, W) float32 [0,1]
        Returns: stacked_obs (3 * len(sliding_window), out_size, out_size)
        """
        processed = []

        for frame_chw in sliding_window:
            # Convert CHW → HWC
            frame_hwc = np.transpose(frame_chw, (1, 2, 0))  # (H, W, 3)

            # Resize to (out_size, out_size)
            resized = cv2.resize(
                frame_hwc,
                (self.out_size, self.out_size),
                interpolation=cv2.INTER_LINEAR
            )

            # Back to CHW
            resized_chw = np.transpose(resized, (2, 0, 1))  # (3, S, S)

            processed.append(resized_chw.astype(np.float32))

        # Stack along channel dimension
        final = np.concatenate(processed, axis=0)  # (3*stack, S, S)

        return final
