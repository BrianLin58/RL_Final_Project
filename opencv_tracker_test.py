import cv2
import os

# ----------------------------
# 0. Settings
# ----------------------------
IMG_DIR = "data/GOT10/train/GOT-10k_Train_000001/degraded"                  # folder containing images
INIT_BBOX = (347, 443, 429, 272)        # (x, y, w, h) on first frame
OUTPUT_FILE = "train_001_degraded_results.txt"         # file to save results

# ----------------------------
# 1. Create tracker
# ----------------------------
tracker = cv2.TrackerCSRT_create()

# ----------------------------
# 2. Load sorted frames
# ----------------------------
frame_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".png"))
])

# ----------------------------
# 3. Initialize on first frame
# ----------------------------
first_frame = cv2.imread(os.path.join(IMG_DIR, frame_files[0]))
tracker.init(first_frame, INIT_BBOX)

# ----------------------------
# 4. Process frames + save results
# ----------------------------
with open(OUTPUT_FILE, "w") as f:
    # Save the first frame's bbox too
    f.write(f"{INIT_BBOX[0]}, {INIT_BBOX[1]}, {INIT_BBOX[2]}, {INIT_BBOX[3]}\n")

    for fname in frame_files[1:]:
        frame = cv2.imread(os.path.join(IMG_DIR, fname))
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = map(int, bbox)
            f.write(f"{x}, {y}, {w}, {h}\n")
        else:
            f.write("tracking_failed\n")
