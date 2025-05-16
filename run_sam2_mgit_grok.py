import os
import numpy as np
import torch
import time
from sam2.build_sam import build_sam2_video_predictor

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SAM2 model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Base paths
data_base_path = "/data1/zzd/VideoCube/MGIT/data/test"
gt_base_path = "/data1/zzd/VideoCube/sam_track_homework/groundtruth"
test_list_path = "/data1/zzd/VideoCube/sam_track_homework/test_list.txt"

# Read test sequences
with open(test_list_path, 'r') as f:
    sequences = [line.strip() for line in f if line.strip()]

print(f"Total sequences to process: {len(sequences)}")

# Create output directories
os.makedirs("tracker/result", exist_ok=True)
os.makedirs("tracker/time", exist_ok=True)

already_processed = set()
already_processed.update({"001", "006", "007", "012", "022", "038", "045", "061", "074", "079", "087", "089", "107"})
fail_processed = set()
fail_processed.update({"093"})

# Process each sequence
for sequence in sequences:
    if sequence in already_processed:
        print(f"Skipping already processed sequence {sequence}")
        continue
    if sequence in fail_processed:
        print(f"Skipping failed sequence {sequence}")
        continue
    print(f"Processing sequence {sequence}")
    
    # Set video directory (assuming frames are directly in sequence folder)
    video_dir = os.path.join(data_base_path, sequence, f"frame_{sequence}")
    
    # Load ground truth and get first bounding box
    gt_file = os.path.join(gt_base_path, f"{sequence}.txt")
    with open(gt_file, 'r') as f:
        first_line = f.readline().strip()
    first_line = first_line.strip()  # Remove leading/trailing whitespace
    xmin, ymin, width, height = map(float, first_line.split(','))  # Split on commas
    box = np.array([xmin, ymin, xmin + width, ymin + height], dtype=np.float32)
    
    # Initialize SAM2 inference state
    inference_state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    )
    
    # Add box prompt for first frame
    ann_frame_idx = 0
    ann_obj_id = 1  # Single object per sequence
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box
    )
    
    # Propagate tracking across the video
    video_segments = {}
    runtimes = []
    prev_time = time.time()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        now = time.time()
        runtimes.append(now - prev_time)
        prev_time = now

        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
        video_segments[out_frame_idx] = mask

    predictor.reset_state(inference_state)  # Reset predictor state
    
    # Convert masks to bounding boxes
    bboxes = []
    for frame_idx in range(len(video_segments)):
        mask = video_segments[frame_idx]
        if mask.any():
            rows, cols = np.where(mask)
            ymin, ymax = rows.min(), rows.max()
            xmin, xmax = cols.min(), cols.max()
            width = xmax - xmin + 1
            height = ymax - ymin + 1
            bbox = [float(xmin), float(ymin), float(width), float(height)]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]  # Default for no detection
        bboxes.append(bbox)
    
    # Save bounding boxes
    result_file = f"tracker/result/tracker_{sequence}.txt"
    with open(result_file, 'w') as f:
        for bbox in bboxes:
            f.write(','.join(map(str, bbox)) + '\n')
    
    # Save runtimes
    time_file = f"tracker/time/tracker_{sequence}.txt"
    with open(time_file, 'w') as f:
        for runtime in runtimes:
            f.write(f"{runtime}\n")
    
    # Clean up inference state
    # predictor.reset_state(inference_state

# Create zip file
os.system("zip -r submission.zip tracker")
print("Submission zip file created: submission.zip")