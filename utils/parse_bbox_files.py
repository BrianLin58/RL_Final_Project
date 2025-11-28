
def parse_bbox_from_files(filepath):
    """
    Parse bounding box coordinates from a text file.

    The function reads a file line-by-line, expecting each line to contain at
    least four numeric values representing bounding box parameters (x, y, w, h).
    Commas are treated as delimiters and converted to spaces.

    Returns:
        tuple:
            - boxes (list of tuple): A list of (x, y, w, h) bounding boxes as integer.
            - init_box (tuple): The first bounding box in the list.

    Notes:
        Lines that contain fewer than four values are skipped.
        Raises IndexError if the file contains no valid bounding boxes.
    """
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.replace(',', ' ').split()

            if len(parts) < 4:
                print(f"Warning: Encountered a line with len {len(parts)}. Skipping...")
                continue

            try:
                # print(f"[DEBUG] x, y, w, h = {parts}")
                x, y, w, h = (int(float(v)) for v in parts[:4])
            except ValueError:
                print(f"Warning: Encountered a line with invalid input. Skipping...")
                # continue

            boxes.append((x, y, w, h))
            # print(f"[DEBUG] Appended {(x, y, w, h)} to box")

    return boxes, boxes[0]