# usage example: python dataset/degrade_video_albu.py --input_root ../dataset/GOT10/val/GOT-10k_Val_000002 --output_root data/GOT10/val/ --make_video
import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import argparse
import shutil
import yaml
import copy

# ===========================
# Helper Functions
# ===========================
def is_sequence_folder(folder):
    """Check if folder directly contains frame images."""
    return any(f.lower().endswith((".jpg", ".png"))
               for f in os.listdir(folder))


def get_sequence_folders(parent):
    """Return all subfolders containing frames."""
    seqs = []
    for f in os.listdir(parent):
        path = os.path.join(parent, f)
        if os.path.isdir(path) and is_sequence_folder(path):
            seqs.append(path)
    return seqs


def relpath_inside(root, path):
    """Return path of sequence relative to root."""
    return os.path.relpath(path, root)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def load_frames_as_video(folder):
    frame_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png"))
    ])
    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(folder, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    try:
        return np.stack(frames), frame_files
    except:
        return None, None


def save_frames(video_array, filenames, out_folder):
    mkdir(out_folder)
    for img, fname in zip(video_array, filenames):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_folder, fname), img_bgr)


def copy_metadata_only(src, dst):
    mkdir(dst)
    skip_ext = {".jpg", ".png", ".mp4"}

    for fname in os.listdir(src):
        src_p = os.path.join(src, fname)
        dst_p = os.path.join(dst, fname)

        if os.path.isdir(src_p):
            continue

        ext = os.path.splitext(fname)[1].lower()
        if ext in skip_ext:
            continue

        shutil.copy2(src_p, dst_p)

def make_video(frames_folder, out_path, fps=30):
    files = sorted(os.listdir(frames_folder))
    files = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
    if not files:
        print("[WARN] No frames to make video.")
        return

    first = cv2.imread(os.path.join(frames_folder, files[0]))
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    for f in tqdm(files, desc=f"[Video] {out_path}"):
        frame = cv2.imread(os.path.join(frames_folder, f))
        writer.write(frame)

    writer.release()
    print(f"[INFO] Video saved: {out_path}")


# =========================================
# CONFIG PARSER
# =========================================
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

'''
# below is deprecated
'
def build_video_transform():
    """
    A flicker-free video degradation pipeline.
    All randomness is sampled ONCE for the whole video.
    """

    return A.Compose([
        A.MotionBlur(blur_limit=37, p=1.0, allow_shifted = False),
        A.GaussianBlur(blur_limit=(0, 0), sigma_limit=(1.0, 2.5), p=1.0),
        A.Downscale(scale_range=[0.1, 0.1], p=1.0),
        A.GaussNoise(mean_range=[-0.2, -0.2], p=1.0),
        A.ImageCompression(quality_range=[10, 40], p=1.0),
    ])
'''
# ===========================
# Degradation Pipeline
# ===========================

def degrade_single_sequence(seq_folder, input_root, output_root, cfg, make_vid = False):
    print(f"[INFO] Processing sequence: {seq_folder}")

    # Mirror directory structure
    relative = relpath_inside(input_root, seq_folder)
    # print(f"[DEBUG] relative = {relative}")
    out_seq_root = os.path.join(output_root, relative)

    degraded_dir = os.path.join(out_seq_root, "degraded")
    original_dir = os.path.join(out_seq_root, "original")

    mkdir(out_seq_root)

    video, frame_files = load_frames_as_video(seq_folder)
    if video is None:
        print(f"[WARNING] Cannot process {seq_folder} (load error)")
        return
    transform = build_transform_from_config(cfg)
    # transform = build_video_transform()
    augmented = transform(images=video)
    degraded = augmented["images"]

    # apply noise
    if "noise" in cfg:
        noise_cfg = cfg["noise"]
        
        # if noise_cfg is a list, take the first element
        if isinstance(noise_cfg, list):
            if len(noise_cfg) == 0:
                return degraded
            noise_cfg = noise_cfg[0]

        noise_type = getattr(A, noise_cfg["type"])
        noise_params = copy.deepcopy(noise_cfg.get("params", {}))

       # deterministic but unique per sequence
        # seq_seed = abs(hash(seq)) % (2**32)
        # np.random.seed(seq_seed)
        gn_mean = np.random.uniform(low = noise_params.get("mean_range", [0.0, 0.0])[0],
                                    high = noise_params.get("mean_range", [0.0, 0.0])[1])
        # print(f"[DEBUG] Sampled gaussian noise mean = {gn_mean} across the video.")
        noise_params["mean_range"] = [gn_mean, gn_mean]

        gn_std = np.random.uniform(low = noise_params.get("std_range", [0.2, 0.44])[0],
                                   high = noise_params.get("std_range", [0.2, 0.44])[1])
        # print(f"[DEBUG] Sampled gaussian noise std = {gn_std} across the video.")
        noise_params["std_range"] = [gn_std, gn_std]

        noise_p = noise_cfg.get("p", 1.0)
        noise_transform = noise_type(p=noise_p, **noise_params)

        # apply per-frame
        degraded = [
            noise_transform(image=frame)["image"]
            for frame in degraded
        ]

    # Save outputs
    save_frames(video, frame_files, original_dir)
    save_frames(degraded, frame_files, degraded_dir)

    # Copy metadata
    copy_metadata_only(seq_folder, out_seq_root)

    print(f"[DONE] Saved to: {out_seq_root}")

    if make_vid:
        make_video(original_dir, os.path.join(os.path.dirname(original_dir), "original.mp4"))
        make_video(degraded_dir, os.path.join(os.path.dirname(original_dir), "degraded.mp4"))


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True,
                        help="Dataset root (train/ or test/ or single sequence)")
    parser.add_argument("--output_root", required=True,
                        help="Output dataset root (will mirror structure)")
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--config_path", default="dataset/default_aug.yaml", type = str)
    parser.add_argument("--mode", required=True, type = str,
                        help = "'single' for one vid, your input_root should be a directory directly containing frame images; \
                            'multiple' for a folder containing multiple videos, your input_root should be a dirctory containing directories of frames; \
                            'txt' mode requires a txt file as argument indicating the relative paths of desired directories to the input_root, the input_root should be a directory containing multiple folders of frames.")
    parser.add_argument("--txt", type = str, default = None)
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    mode = args.mode

    cfg = load_config(args.config_path)

    full_path = []
    root = None

    if mode == 'single':
        full_path.append(input_root)
        root = os.path.dirname(os.path.normpath(input_root))
        # print(f"[DEBUG] full_path = {full_path}, root = {root}")
    elif mode == 'multiple':
        full_path = get_sequence_folders(input_root)
        root = input_root
        if len(full_path) == 0:
            print("[ERROR] No sequences found.")
            exit(1)
        print(f"[INFO] Found {len(full_path)} sequences.")
    elif mode == 'txt':
        if args.txt is None:
            print('[ERROR] You have to specify your txt file with txt mode.')
            exit(1)
        with open(args.txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    full_path.append(os.path.join(input_root, line))
        print(f"[INFO] Found {len(full_path)} sequences.")
        root = input_root

    for seq in full_path:
        # print(f"[DEBUG] seq = {seq}, root = {root}")
        degrade_single_sequence(seq, root, output_root, cfg, args.make_video)
    print("[INFO] Finished degrading videos.")

    # # Case 1: Direct sequence folder
    # if is_sequence_folder(input_root):
    #     # print(f"[DEBUG] input_root = {input_root}, dirname = {os.path.dirname(input_root)}, output_root = {output_root}")
    #     degrade_single_sequence(input_root, os.path.dirname(input_root), output_root, cfg, args.make_video)
    #     exit()

    # # Case 2: Folder of many sequences
    # seqs = get_sequence_folders(input_root)
    # if len(seqs) == 0:
    #     print("[ERROR] No sequences found.")
    #     exit(1)

    # print(f"[INFO] Found {len(seqs)} sequences.")
    # for seq in seqs:
    #     degrade_single_sequence(seq, input_root, output_root, cfg, args.make_video)
