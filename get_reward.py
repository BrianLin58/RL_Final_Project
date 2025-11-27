import os

from pytracking.run_frames import track_folder
from calc_similarity import calc_miou


def evaluate_sequence_miou(seq_dir,
                           tracker_name='trdimp',
                           tracker_param='trdimp',
                           fps=30.0,
                           index=None):
    """
    seq_dir: folder containing 3–120 frames (0000001.jpg, ...) under e.g. .../original
    and a groundtruth.txt in the parent directory.

    index: int or None
        If not None, only the first `index` frames are used when computing mIoU.
        This should match the number of frames you actually track (3–120).

    returns: mean IoU between groundtruth and tracker prediction
    """
    # run tracking on frames
    pred_path = track_folder(
        frames_dir=seq_dir,
        tracker_name=tracker_name,
        tracker_param=tracker_param,
        debug=0,
        save_results=True,
        fps=fps,
    )

    # ground-truth file in the parent folder
    parent = os.path.dirname(os.path.abspath(seq_dir))
    gt_path = os.path.join(parent, 'groundtruth.txt')

    # compute mIoU, restricted to `index` frames if provided
    miou = calc_miou(gt_path, pred_path, num_frames=index)
    return miou


if __name__ == "__main__":
    seq_dir = "data/GOT10/train/GOT-10k_Train_000001/original"
    reward = evaluate_sequence_miou(seq_dir, index=None)
    print("Reward (mIoU):", reward)
