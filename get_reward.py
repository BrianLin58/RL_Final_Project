import os
import sys

from pytracking.run_frames import track_folder
from calc_similarity import calc_miou

def evaluate_sequence_miou(seq_dir,
                           tracker_name='trdimp',
                           tracker_param='trdimp',
                           fps=30.0):
    """
    seq_dir: folder containing 0000001.jpg ... 0000120.jpg and groundtruth.txt
    returns: mean IoU between groundtruth and tracker prediction
    """
    # run tracking
    pred_path = track_folder(
        frames_dir=seq_dir,
        tracker_name=tracker_name,
        tracker_param=tracker_param,
        debug=0,
        save_results=True,
        fps=fps,
    )

    # ground-truth file in the same folder
    # gt_path = os.path.join(seq_dir, 'groundtruth.txt')

    # ground-truth file in the parent folder
    parent = os.path.dirname(os.path.abspath(seq_dir))
    gt_path = os.path.join(parent, 'groundtruth.txt')

    

    # compute mIoU
    miou = calc_miou(gt_path, pred_path)
    return miou


if __name__ == "__main__":
    seq_dir = "data/GOT10/train/GOT-10k_Train_000001/original"
    reward = evaluate_sequence_miou(seq_dir)
    print("Reward (mIoU):", reward)
