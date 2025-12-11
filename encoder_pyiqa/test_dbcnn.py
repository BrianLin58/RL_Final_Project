import argparse
import torch
import pyiqa
import cv2
import os


def load_image_as_tensor(img_path, device):
    """Load an image (HWC, BGR) -> tensor (1, 3, H, W) in [0,1] on device."""
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
        help="Path to test image (degraded or clean).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Create DBCNN metric
    print("[INFO] Loading DBCNN model from pyiqa...")
    metric = pyiqa.create_metric("dbcnn", device=device, as_loss=False)
    metric.eval()

    # Option A: pass tensor
    img_tensor = load_image_as_tensor(args.image, device)

    with torch.no_grad():
        score_tensor = metric(img_tensor)  # NR metric: single input
    score = float(score_tensor.item())

    print(f"[RESULT] DBCNN quality score for {args.image}: {score:.4f}")

    # Option B (optional): call with path directly as in some pyiqa examples
    # This is just to test that both interfaces work.
    with torch.no_grad():
        score_from_path = metric(args.image)
    print(f"[RESULT] DBCNN score using path input: {float(score_from_path.item()):.4f}")

    # Show model structure briefly (to inspect later for features)
    # print("\n[INFO] Model structure (truncated):")
    # print(metric)


if __name__ == "__main__":
    main()
