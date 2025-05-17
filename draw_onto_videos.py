import os
import cv2
import numpy as np

# Define paths
video_folder = "/data1/zzd/VideoCube/MGIT_test_mp4"  # Adjust to your MP4 folder
bbox_folder = "tracker/result"
output_folder = "annotated_videos"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Get list of MP4 files
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
video_files.sort()  # Sort files to maintain order
for video_file in video_files:
    # Extract sequence ID (e.g., "001" from "001.mp4")
    sequence_id = os.path.splitext(video_file)[0]
    print(f"Processing video: {video_file} (Sequence {sequence_id})")
    
    # Paths
    video_path = os.path.join(video_folder, video_file)
    bbox_file = os.path.join(bbox_folder, f"tracker_{sequence_id}.txt")
    output_video = os.path.join(output_folder, f"annotated_{sequence_id}.mp4")
    
    # Check if bbox file exists
    if not os.path.exists(bbox_file):
        print(f"Bounding box file {bbox_file} not found, skipping.")
        continue
    
    # Read bounding boxes
    bboxes = []
    with open(bbox_file, 'r') as f:
        for line in f:
            try:
                xmin, ymin, width, height = map(float, line.strip().split(','))
                bboxes.append([xmin, ymin, width, height])
            except ValueError:
                print(f"Invalid bbox format in {bbox_file}, line: {line.strip()}")
                bboxes.append([0, 0, 0, 0])  # Fallback for invalid lines
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}, skipping.")
        continue
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Verify bbox count matches frame count
    if len(bboxes) != frame_count:
        print(f"Warning: {len(bboxes)} bboxes for {frame_count} frames in {video_file}")
    
    # Initialize temporary output video writer (raw video, no compression)
    temp_output = os.path.join(output_folder, f"temp_{sequence_id}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID for temporary uncompressed output
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get bbox for current frame (use last bbox if out of bounds)
        bbox = bboxes[min(frame_idx, len(bboxes) - 1)]
        xmin, ymin, width, height = bbox
        
        # Draw bbox if valid (non-zero width/height)
        if width > 0 and height > 0:
            top_left = (int(xmin), int(ymin))
            bottom_right = (int(xmin + width), int(ymin + height))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green, 2px thick
        
        # Write frame to temporary video
        out.write(frame)
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Convert temporary AVI to MP4 using FFmpeg for efficiency
    ffmpeg_cmd = (
        f"ffmpeg -y -i {temp_output} -c:v libx264 -preset fast -crf 22 "
        f"-c:a aac -b:a 128k {output_video}"
    )
    os.system(ffmpeg_cmd)
    
    # Remove temporary AVI file
    os.remove(temp_output)
    
    print(f"Annotated video saved: {output_video}")

print("All videos processed.")