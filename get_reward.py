import cv2
from calc_similarity import calc_miou_from_boxes


# --------------------------------------------------------
# Track directly on list of NumPy frames
# --------------------------------------------------------
def run_opencv_tracker_on_frames(frames, init_bbox):
    """
    frames: list of numpy HxWxC images (RGB or BGR)
    init_bbox: (x, y, w, h)
    returns: list of predicted boxes [(x,y,w,h), ...]
    """

    assert len(frames) > 0, "No frames provided."
    tracker = cv2.TrackerCSRT_create()

    # Ensure BGR (OpenCV uses BGR)
    first_frame = frames[0]
    if first_frame.shape[-1] == 3:
        pass  # assume already BGR
    else:
        raise ValueError("Frame must be HxWx3")

    tracker.init(first_frame, tuple(init_bbox))

    pred_boxes = [tuple(init_bbox)]

    # Track forward
    for frame in frames[1:]:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(float, bbox)
            pred_boxes.append((x, y, w, h))
        else:
            # failed: append zero-box or copy previous box
            pred_boxes.append((0, 0, 0, 0))

    return pred_boxes


# --------------------------------------------------------
# Compute mIoU directly from frames + GT boxes
# --------------------------------------------------------
def evaluate_sequence_miou_from_frames(frames, gt_boxes, index=None):
    """
    frames: list of np.ndarray images
    gt_boxes: list of (x,y,w,h) ground truth for each frame
    index: number of frames to evaluate (truncate)

    Returns: mean IoU
    """

    # The first bbox is used to initialize tracker
    init_bbox = gt_boxes[0]
    init_bbox = [int(v) for v in init_bbox]     # mutable list
    # print(f"[DEBUG] init_bbox = {init_bbox} with type: {type(init_bbox[0])}")

    # Run tracker directly in memory
    # for i, f in enumerate(frames):
        # print(f"[DEBUG] i, f.shape = {i}, {f.shape}")

    pred_boxes = run_opencv_tracker_on_frames(frames, init_bbox)

    # Compute IoU
    miou = calc_miou_from_boxes(gt_boxes, pred_boxes, num_frames=index)
    return miou

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
# Simple unit test
# --------------------------------------------------------
if __name__ == "__main__":
    import cv2
    import glob

    # Example: load frames manually just for test
    frame_paths = sorted(glob.glob("data/GOT10/train/GOT-10k_Train_000001/original/*.jpg"))
    frames = [cv2.imread(p) for p in frame_paths]

    # Example ground truth
    gt_boxes = []
    with open("data/GOT10/train/GOT-10k_Train_000001/groundtruth.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace(',', ' ')
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, w, h = map(float, parts[:4])
            gt_boxes.append((x, y, w, h))

    miou = evaluate_sequence_miou_from_frames(frames, gt_boxes)
    print("mIoU =", miou)
