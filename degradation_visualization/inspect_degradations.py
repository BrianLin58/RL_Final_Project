import os
import cv2
import yaml
import numpy as np
import albumentations as A
import argparse


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_transform_from_config(cfg):
    aug_list = []
    for aug in cfg["augmentations"]:
        aug_type = getattr(A, aug["type"])
        params = aug.get("params", {})
        p = aug.get("p", 1.0)
        aug_list.append(aug_type(p=p, **params))
    return A.Compose(aug_list)


def tile_images(images, pad=5):
    """Tile images horizontally with padding."""
    h, w, c = images[0].shape
    canvas = np.ones((h, w * len(images) + pad * (len(images) - 1), c),
                     dtype=images[0].dtype) * 255
    x = 0
    for img in images:
        canvas[:, x:x + w, :] = img
        x += w + pad
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a test image")
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--out", default="inspect_result.jpg", help="Output image path")
    parser.add_argument("--n", type=int, default=4, help="Number of augmented samples")
    args = parser.parse_args()

    # Load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Load config and build transform
    cfg = load_config(args.cfg)
    transform = build_transform_from_config(cfg)

    samples = []
    # Include original first
    samples.append(img_rgb)
    for i in range(args.n):
        augmented = transform(image=img_rgb)
        samples.append(augmented["image"])

    # Tile and save
    grid_rgb = tile_images(samples, pad=8)
    grid_bgr = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, grid_bgr)
    print(f"[INFO] Saved inspection image to {args.out}")
    print("[INFO] Left-most = original, others = augmented samples")


if __name__ == "__main__":
    main()
