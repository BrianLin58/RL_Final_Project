import torch
import torch.nn as nn
from dbcnn_feature_wrapper import DbcnnEncoder

class DBCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # X1+X2 -> 640-D feature
        self.encoder = DbcnnEncoder(use_x1=True, use_x2=True)

    def forward(self, x):
        """
        x: torch.Tensor, shape (B, 3, H, W), float32 in [0,1]
        return: (B, 640)
        """
        return self.encoder(x)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = DBCNNEncoder().to(device)
    # Suppose we have 3 RGB images of size 1080x1920 in [0,1]
    dummy_input = torch.rand(3, 3, 1080, 1920, device=device)  # [0,1]

    with torch.no_grad():
        output = encoder(dummy_input)

    print(output.shape)  # torch.Size([3, 640]) if use_x1+use_x2
