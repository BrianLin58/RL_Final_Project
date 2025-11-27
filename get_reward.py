import os
import cv2
from calc_similarity import calc_miou


# --------------------------------------------------------
# OpenCV tracker for a folder of frames
# --------------------------------------------------------
def run_opencv_tracker(frames_dir, init_bbox, output_file="pred.txt"):
    """
    frames_dir: path to folder containing frames (00000001.jpg, ...)
    init_bbox: initial bbox in (x, y, w, h)
    output_file: txt file to save predictions

    Returns: path to output_file
    """

    tracker = cv2.TrackerCSRT_create()      # you can change tracker type
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    assert len(frame_files) > 0, "No frames found."

    # Init tracker on first frame
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    tracker.init(first_frame, tuple(init_bbox))

    # Save results
    out_path = os.path.join(frames_dir, output_file)
    with open(out_path, "w") as f:
        # first bbox
        f.write(f"{init_bbox[0]}, {init_bbox[1]}, {init_bbox[2]}, {init_bbox[3]}\n")

        # update for all other frames
        for fname in frame_files[1:]:
            frame = cv2.imread(os.path.join(frames_dir, fname))
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = map(int, bbox)
                f.write(f"{x}, {y}, {w}, {h}\n")
            else:
                f.write("tracking_failed\n")

    return out_path


# --------------------------------------------------------
# High-level evaluation pipeline
# --------------------------------------------------------
def evaluate_sequence_miou(seq_dir, gt_path, index=None):
    """
    seq_dir: folder containing frames (0001.jpg, ...) under "original/"
    init_bbox: (x, y, w, h)
    index: limit number of frames when computing mIoU

    Returns: mean IoU
    """

    # 0. Get bbox
    # parent = os.path.dirname(os.path.abspath(seq_dir))
    # parent = os.path.dirname(os.path.abspath(parent))
    # gt_path = os.path.join(parent, "groundtruth.txt")
    with open(gt_path, 'r') as f:
        line = f.readline().strip()

    # handle "347.0000,443.0000,429.0000,272.0000" or space separated
    for sep in [',', ' ']:
        parts = [p for p in line.replace(' ', '').split(sep) if p]
        if len(parts) == 4:
            init_bbox = [int(float(p)) for p in parts]

    # 1. Run OpenCV tracker → pred.txt
    pred_path = run_opencv_tracker(
        frames_dir=seq_dir,
        init_bbox=init_bbox,
        output_file="pred.txt",
    )

    # 2. Ground truth file is in parent folder
    parent = os.path.dirname(os.path.abspath(seq_dir))

    # 3. Compute mIoU
    miou = calc_miou(gt_path, pred_path, num_frames=index)
    return miou


# --------------------------------------------------------
# Main test
# --------------------------------------------------------
if __name__ == "__main__":
    seq_dir = "data/GOT10/train/GOT-10k_Train_000001/original" # test

    reward = evaluate_sequence_miou(seq_dir, os.path.dirname(seq_dir), index=None)
    print("Reward (mIoU):", reward)
