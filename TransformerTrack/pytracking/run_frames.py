import argparse
import os
import sys
import glob
import tempfile

# env_path = repo root (../ from pytracking/)
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import cv2
from pytracking.evaluation import Tracker


def frames_to_video(frames_dir, fps=30.0, fourcc_str='MJPG'):
    """Pack all images in frames_dir into a temporary video file and return its path."""
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    frame_paths = []
    for ext in exts:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, ext)))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No image frames found in {frames_dir}")

    frame_paths = sorted(frame_paths)

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    # name tmp video after the folder (GOT-10k_Train_000001.avi)
    seq_name = os.path.basename(os.path.normpath(frames_dir))
    tmp_dir = tempfile.gettempdir()
    video_path = os.path.join(tmp_dir, f"{seq_name}.avi")

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Check codec support.")

    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {fp}")
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    return video_path


def load_init_box_from_gt(frames_dir):
    """Read the FIRST line of groundtruth.txt as [x, y, w, h]."""
    # gt_path = os.path.join(frames_dir, 'groundtruth.txt')

    parent = os.path.dirname(os.path.abspath(frames_dir))
    gt_path = os.path.join(parent, 'groundtruth.txt')

    if not os.path.exists(gt_path):
        raise RuntimeError(f'groundtruth.txt not found in {frames_dir}')

    with open(gt_path, 'r') as f:
        line = f.readline().strip()

    # handle "347.0000,443.0000,429.0000,272.0000" or space separated
    for sep in [',', ' ']:
        parts = [p for p in line.replace(' ', '').split(sep) if p]
        if len(parts) == 4:
            return [float(p) for p in parts]

    raise RuntimeError(f'Could not parse first line of groundtruth.txt: {line}')


def track_folder(frames_dir,
                 tracker_name='trdimp',
                 tracker_param='trdimp',
                 debug=0,
                 save_results=True,
                 fps=30.0):
    """
    Run tracker on a folder of frames and RETURN the prediction txt path.

    Returns
    -------
    pred_txt_path : str
        pytracking/tracking_results/<tracker_name>/<tracker_param>/video_<seq_name>.txt
    """
    # 1) frames -> temp video
    video_path = frames_to_video(frames_dir, fps=fps)
    print(f"[run_frames] Created temporary video: {video_path}")

    # 2) init box from groundtruth.txt
    init_box = load_init_box_from_gt(frames_dir)

    # 3) run tracker
    tracker = Tracker(tracker_name, tracker_param)
    tracker.run_video(videofilepath=video_path,
                      optional_box=init_box,
                      debug=debug,
                      save_results=save_results)

    # 4) construct where run_video stored the result
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    pred_txt_path = os.path.join(
        env_path,
        'pytracking',
        'tracking_results',
        tracker_name,
        tracker_param,
        f'video_{video_name}.txt'
    )
    return pred_txt_path


def main():
    parser = argparse.ArgumentParser(description='Run tracker on a folder of frames.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('frames_dir', type=str, help='Folder containing frames.')
    parser.add_argument('--debug', type=int, default=0,
                        help='Debug level.')
    parser.add_argument('--save_results', action='store_true',
                        help='Save tracking results.')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='FPS to use when packing frames into a video.')
    args = parser.parse_args()

    pred_txt = track_folder(
        frames_dir=args.frames_dir,
        tracker_name=args.tracker_name,
        tracker_param=args.tracker_param,
        debug=args.debug,
        save_results=args.save_results,
        fps=args.fps,
    )
    print("[run_frames] Prediction saved to:", pred_txt)


if __name__ == '__main__':
    main()
