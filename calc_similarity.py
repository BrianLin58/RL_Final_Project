import sys


def parse_boxes(path):
    """
    Read a file where each line is x, y, w, h.
    """
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace(',', ' ')
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, w, h = map(float, parts[:4])
            boxes.append((x, y, w, h))
    return boxes


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_br, y1_br = x1 + w1, y1 + h1
    x2_br, y2_br = x2 + w2, y2 + h2

    xi_left   = max(x1, x2)
    yi_top    = max(y1, y2)
    xi_right  = min(x1_br, x2_br)
    yi_bottom = min(y1_br, y2_br)

    inter_w = max(0.0, xi_right - xi_left)
    inter_h = max(0.0, yi_bottom - yi_top)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


# ----------------------------------------------------------
# NEW: compute mIoU directly from box lists (no file I/O)
# ----------------------------------------------------------
def calc_miou_from_boxes(gt_boxes, pred_boxes, num_frames=None):
    """
    Compute mean IoU directly from lists of (x,y,w,h) tuples.
    gt_boxes : list of (x,y,w,h)
    pred_boxes : list of (x,y,w,h)
    """

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        raise ValueError("No boxes to compare.")

    if num_frames is None:
        if len(gt_boxes) != len(pred_boxes):
            raise ValueError(
                f"Different lengths: {len(gt_boxes)} vs {len(pred_boxes)}"
            )
        n = len(gt_boxes)
    else:
        n = min(num_frames, len(gt_boxes), len(pred_boxes))

    # ious = [iou(gt_boxes[i], pred_boxes[i]) for i in range(n)]
    ious = []
    for i in range(n):
        single_iou = iou(gt_boxes[i], pred_boxes[i])
        # print(f"The {i}th frame has iou {single_iou}")
        ious.append(single_iou)
    return float(sum(ious) / n)

def calc_weighted_miou_from_boxes(gt_boxes, pred_boxes, num_frames=None, decay_factor=0.5, chunk_size=32, first_chunk_size=34):
    """
    Compute weighted mean IoU with exponential decay for older frames.
    
    Chunking pattern: [34, 32, 32, ..., 32, remainder]
    - First chunk (oldest): Always 34 frames (if n >= 34)
    - Middle chunks: All 32 frames each
    - Last chunk (newest): Remainder (1 to 32 frames, has highest weight)
    
    Weighting (newest frames have highest weight):
    - Last chunk (newest): weight = 1.0
    - Second-to-last chunk: weight = decay_factor
    - Third-to-last chunk: weight = decay_factor^2
    - And so on...
    
    Examples:
    - 20 frames: [20] (only 1 chunk, weight=1.0)
    - 34 frames: [34] (only 1 chunk, weight=1.0)
    - 64 frames: [34, 30] (first=34, last=30 with weight=1.0)
    - 66 frames: [34, 32] (first=34 weight=0.5, last=32 weight=1.0)
    - 70 frames: [34, 32, 4] (first=34 weight=0.25, middle=32 weight=0.5, last=4 weight=1.0)
    - 98 frames: [34, 32, 32] (first=34 weight=0.25, two middles=32 each, last weight=1.0)
    - 100 frames: [34, 32, 32, 2] (first=34, two middles=32, last=2 with weight=1.0)
    
    Args:
        gt_boxes: list of (x,y,w,h) ground truth boxes
        pred_boxes: list of (x,y,w,h) predicted boxes
        num_frames: optional number of frames to compare (default: all)
        decay_factor: weight multiplier for each older chunk (default: 0.5)
        chunk_size: size of middle chunks (default: 32)
        first_chunk_size: size of first (oldest) chunk (default: 34)
    
    Returns:
        Weighted mean IoU as float
    """
    
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        raise ValueError("No boxes to compare.")

    if num_frames is None:
        if len(gt_boxes) != len(pred_boxes):
            raise ValueError(
                f"Different lengths: {len(gt_boxes)} vs {len(pred_boxes)}"
            )
        n = len(gt_boxes)
    else:
        n = min(num_frames, len(gt_boxes), len(pred_boxes))

    # Calculate IoU for all frames
    ious = []
    for i in range(n):
        single_iou = iou(gt_boxes[i], pred_boxes[i])
        ious.append(single_iou)
    
    # Build chunk boundaries from old to new
    # Pattern: [34, 32, 32, ..., 32, remainder]
    chunks = []
    start_idx = 0
    
    if n <= first_chunk_size:
        # Only one chunk if not enough frames
        chunks.append((start_idx, n))
    else:
        # First chunk: always first_chunk_size (34)
        chunks.append((start_idx, start_idx + first_chunk_size))
        start_idx += first_chunk_size
        
        remaining = n - first_chunk_size
        
        # Middle chunks: all chunk_size (32) each
        num_middle_chunks = remaining // chunk_size
        for _ in range(num_middle_chunks):
            chunks.append((start_idx, start_idx + chunk_size))
            start_idx += chunk_size
        
        # Last chunk: remainder (1 to chunk_size frames)
        # If remainder is 0, there's no separate last chunk (last middle chunk IS the last)
        remainder = remaining % chunk_size
        if remainder > 0:
            chunks.append((start_idx, start_idx + remainder))
    
    # Calculate weighted sum
    # Chunks are ordered from oldest to newest
    # Weight increases exponentially: oldest gets decay^(k-1), newest gets decay^0 = 1
    num_chunks = len(chunks)
    weighted_sum = 0.0
    total_weight = 0.0
    
    for chunk_idx, (start, end) in enumerate(chunks):
        # Weight for this chunk: decay_factor^(num_chunks - 1 - chunk_idx)
        # Last chunk (chunk_idx = num_chunks-1) gets weight = 1.0
        # Second-to-last gets decay_factor^1, etc.
        weight = decay_factor ** (num_chunks - 1 - chunk_idx)
        
        # Sum IoUs in this chunk
        chunk_iou_sum = sum(ious[start:end])
        chunk_size_actual = end - start
        
        # Add weighted contribution
        weighted_sum += weight * chunk_iou_sum
        total_weight += weight * chunk_size_actual
    
    # Return weighted average
    if total_weight == 0:
        return 0.0
    
    return float(weighted_sum / total_weight)

# ----------------------------------------------------------
# OLD interface: still load files, but uses calc_miou_from_boxes()
# ----------------------------------------------------------
def calc_miou(gt_path, pred_path, num_frames=None):
    gt_boxes = parse_boxes(gt_path)
    pred_boxes = parse_boxes(pred_path)
    return calc_miou_from_boxes(gt_boxes, pred_boxes, num_frames)


# ------------------------------
# CLI tool (unchanged)
# ------------------------------
def main(gt_path, pred_path, num_frames=None):
    if num_frames is None:
        mean_iou = calc_miou(gt_path, pred_path)
    else:
        mean_iou = calc_miou(gt_path, pred_path, num_frames=num_frames)

    print(f"Compared {len(parse_boxes(gt_path)) if num_frames is None else num_frames} frames")
    print(f"Mean IoU similarity: {mean_iou:.4f}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(f"Usage: python {sys.argv[0]} groundtruth.txt video_xxx.txt [num_frames]")
    else:
        gt = sys.argv[1]
        pred = sys.argv[2]
        nf = int(sys.argv[3]) if len(sys.argv) == 4 else None
        main(gt, pred, nf)
