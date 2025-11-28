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

    ious = [iou(gt_boxes[i], pred_boxes[i]) for i in range(n)]
    return float(sum(ious) / n)


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
