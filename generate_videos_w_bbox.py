import os
import cv2
import numpy as np
import subprocess
import shutil
from pathlib import Path

# Base paths
data_base_path = "/data1/zzd/VideoCube/MGIT/data/test"
result_base_path = "tracker/result"
test_list_path = "/data1/zzd/VideoCube/sam_track_homework/test_list.txt"
output_base_path = "visualizations"
temp_frames_path = "temp_frames"

# Create output and temporary directories
os.makedirs(output_base_path, exist_ok=True)
os.makedirs(temp_frames_path, exist_ok=True)

# Read test sequences
with open(test_list_path, 'r') as f:
    sequences = [line.strip() for line in f if line.strip()]

# Process each sequence
for sequence in sequences:
    print(f"Processing sequence {sequence}")
    
    # Paths
    video_dir = os.path.join(data_base_path, sequence, f"frame_{sequence}")  # Adjust if frames are in frame_001/
    bbox_file = os.path.join(result_base_path, f"tracker_{sequence}.txt")
    output_video = os.path.join(output_base_path, f"tracker_{sequence}.mp4")
    
    # Read bounding boxes
    bboxes = []
    with open(bbox_file, 'r') as f:
        for line in f:
            xmin, ymin, width, height = map(float, line.strip().split(','))
            bboxes.append((xmin, ymin, width, height))
    
    # Get frame files
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # Verify frame and bbox count match
    if len(frame_names) != len(bboxes):
        print(f"Warning: Sequence {sequence} has {len(frame_names)} frames but {len(bboxes)} bboxes")
        continue
    
    # Clear temporary frames directory
    shutil.rmtree(temp_frames_path, ignore_errors=True)
    os.makedirs(temp_frames_path, exist_ok=True)
    
    # Process each frame
    for idx, (frame_name, bbox) in enumerate(zip(frame_names, bboxes)):
        frame_path = os.path.join(video_dir, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        # Draw bounding box
        xmin, ymin, width, height = map(int, bbox)
        if width > 0 and height > 0:  # Only draw valid bboxes
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Save annotated frame
        temp_frame_path = os.path.join(temp_frames_path, f"frame_{idx:05d}.jpg")
        cv2.imwrite(temp_frame_path, img)
    
    # Create video with ffmpeg
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-framerate", "30",  # Assume 30 FPS; adjust if known
        "-i", os.path.join(temp_frames_path, "frame_%05d.jpg"),
        "-vf", "pad=iw+mod(iw\,2):ih+mod(ih\,2)",  # Pad to even dimensions
        "-c:v", "libx264",  # Use H.264 codec
        "-preset", "fast",  # Fast encoding
        "-crf", "23",  # Quality (lower is better, 23 is default)
        output_video
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video saved: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video for sequence {sequence}: {e.stderr.decode()}")
    
    # Clean up temporary frames
    shutil.rmtree(temp_frames_path, ignore_errors=True)

print("Visualization complete.")