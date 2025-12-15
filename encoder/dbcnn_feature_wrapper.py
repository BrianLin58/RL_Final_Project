import torch
import torch.nn as nn
import torch.nn.functional as F

from pyiqa.archs.dbcnn_arch import DBCNN  # from pyiqa


class DBCNNWithFeature(DBCNN):
    """
    Extend DBCNN to expose compact GAP-based features instead of 65536-D.
    """

    def _extract_backbone_features(self, X):
        """Get X1, X2 feature maps with matched spatial size."""
        X = self.preprocess(X)  # ImageNet mean/std inside DBCNN

        X1 = self.features1(X)  # VGG-16 branch: (N, 512, H1, W1)
        X2 = self.features2(X)  # SCNN branch:  (N, 128, H2, W2)

        N, _, H, W = X1.shape
        N2, _, H2, W2 = X2.shape
        assert N == N2, "Batch size mismatch between branches"

        if (H != H2) or (W != W2):
            X2 = F.interpolate(X2, (H, W), mode="bilinear", align_corners=True)

        return X1, X2

    def forward_gap_feature(self, X, use_x1=True, use_x2=True):
        """
        Returns:
            score: (N, 1) DBCNN scalar score (regular IQA output)
            feat_small: (N, D_small) compact feature:
                - use_x1=True, use_x2=True  -> 640-D (512+128)
                - use_x1=False, use_x2=True -> 128-D (SCNN only)
        """
        assert use_x1 or use_x2, "At least one of use_x1/use_x2 must be True."

        # X1, X2 = self._extract_backbone_features(X)
        X2 = self.features2(self.preprocess(X))

        # 1) Compute original DBCNN score (bilinear pooling), mainly for sanity
        # N, C1, H, W = X1.shape
        N, C2, H, W = X2.shape

        # X1_flat = X1.view(N, C1, H * W)
        X2_flat = X2.view(N, C2, H * W)
        # Xb = torch.bmm(X1_flat, X2_flat.transpose(1, 2)) / (H * W)
        # Xb = Xb.view(N, C1 * C2)

        # feat_big = torch.sqrt(Xb + 1e-8)
        # feat_big = F.normalize(feat_big)
        # score = self.fc(feat_big)  # (N, 1)

        # 2) Build compact GAP feature
        feat_parts = []
        # if use_x1:
        #     gap1 = F.adaptive_avg_pool2d(X1, 1).view(N, C1)  # (N, 512)
        #     feat_parts.append(gap1)
        if use_x2:
            gap2 = F.adaptive_avg_pool2d(X2, 1).view(N, C2)  # (N, 128)
            feat_parts.append(gap2)

        feat_small = torch.cat(feat_parts, dim=1)  # (N, 640 or 128)
        return None, feat_small


class DBCNNEncoder(nn.Module):
    """
    Drop-in encoder for RL: outputs a fixed-size DBCNN feature per frame.

    default: X1+X2 -> 640-D feature
    """

    def __init__(self, use_x1=True, use_x2=True):
        super().__init__()
        self.dbcnn = DBCNNWithFeature(pretrained=True)
        self.use_x1 = use_x1
        self.use_x2 = use_x2

        # Freeze parameters
        for p in self.dbcnn.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        x: (B, 3, H, W) float32 in [0,1], RGB
        return: (B, D) where D=640 if use_x1+use_x2, or 128 if only use_x2
        """
        with torch.no_grad():
            _, feat_small = self.dbcnn.forward_gap_feature(
                x, use_x1=self.use_x1, use_x2=self.use_x2
            )
        return feat_small
