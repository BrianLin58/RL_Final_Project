#!/usr/bin/env python
import argparse
import os

import cv2
import torch

from dbcnn_feature_wrapper import DBCNNWithFeature


def load_image_as_tensor(img_path, device):
    """
    Load an image from disk and convert to tensor (1, 3, H, W) in [0,1], RGB.
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2.imread failed for {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype("float32") / 255.0  # [0,1]

    # HWC -> CHW
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a test image (clean or degraded).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using device: {device}")

    # Instantiate DBCNN with pretrained weights
    # pretrained=True -> use default 'koniq' weights in dbcnn_arch.py
    print("[INFO] Loading DBCNNWithFeature (pretrained)...")
    net = DBCNNWithFeature(pretrained=True).to(device).eval()

    # Load image
    img = load_image_as_tensor(args.image, device)
    print(f"[INFO] Loaded image {args.image} with shape {tuple(img.shape)}")

    # Forward pass
    with torch.no_grad():
        score, feat = net.forward_with_feature(img)
        score, feat_x2_only = net.forward_gap_feature(img, use_x1=False, use_x2=True)
        score, feat_x1_x2 = net.forward_gap_feature(img, use_x1=True, use_x2=True)

    score_val = float(score.squeeze().item())
    print(f"[RESULT] DBCNN quality score: {score_val:.4f}")
    print(f"[RESULT] Feature shape: {tuple(feat.shape)}")
    print(f"[RESULT] Feature (x2 only) shape: {tuple(feat_x2_only.shape)}")
    print(f"[RESULT] Feature (x1 + x2) shape: {tuple(feat_x1_x2.shape)}")

    # Optional: some quick stats on the feature vector
    feat_cpu = feat.squeeze(0).cpu()
    print(
        f"[RESULT] Feature stats: mean={feat_cpu.mean():.4f}, "
        f"std={feat_cpu.std():.4f}, min={feat_cpu.min():.4f}, "
        f"max={feat_cpu.max():.4f}"
    )
    print(
        f"[RESULT] Feature (x2 only) stats: mean={feat_x2_only.squeeze(0).cpu().mean():.4f}, "
        f"std={feat_x2_only.squeeze(0).cpu().std():.4f}"
    )
    print(
        f"[RESULT] Feature (x1 + x2) stats: mean={feat_x1_x2.squeeze(0).cpu().mean():.4f}, "
        f"std={feat_x1_x2.squeeze(0).cpu().std():.4f}"
    )


if __name__ == "__main__":
    main()
