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

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final fully-connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 512, 1, 1)

        # Freeze encoder parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: torch.Tensor, shape (B, 3, H, W)
        return: (B, 512)
        """
        features = self.feature_extractor(x)      # (B, 512, 1, 1)
        features = torch.flatten(features, 1)     # (B, 512)
        return features

# Example usage
if __name__ == "__main__":
    encoder = ResNet18Encoder()
    # Suppose we have 3 RGB images of size 1080x1920
    dummy_input = torch.randn(3, 3, 1080, 1920)
    output = encoder(dummy_input)
    print(output.shape)  # torch.Size([3, 512])
