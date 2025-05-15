import os
import numpy as np
import torch
import time
from sam2.build_sam import build_sam2_video_predictor

# —— CONFIG ——  
DATA_ROOT     = "/home/dell/sda/datasets/VideoCube/MGIT/data/test"
GT_ROOT       = "/home/dell/sda/datasets/VideoCube/sam_track_homework/groundtruth"
TEST_LIST_TXT = "/home/dell/sda/datasets/VideoCube/sam_track_homework/test_list.txt"
OUT_ROOT      = "./tracker"          # will create result/ under here

SAM2_CKPT     = "checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CFG      = "configs/sam2.1/sam2.1_hiera_t.yaml"

os.makedirs(os.path.join(OUT_ROOT, "result"), exist_ok=True)


# —— BUILD PREDICTOR ——  
device = (
    torch.device("cuda")   if torch.cuda.is_available() else
    torch.device("mps")    if torch.backends.mps.is_available() else
    torch.device("cpu")
)
predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)


# —— UTILITIES ——  
def mask_to_xywh(mask: np.ndarray):
    """Given a 2D bool mask, return x,y,w,h."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0,0,0,0
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1 - x0, y1 - y0


# —— MAIN LOOP ——  
with open(TEST_LIST_TXT, "r") as f:
    seqs = [l.strip().zfill(3) for l in f if l.strip()]

for seq in seqs:
    print(f"Processing sequence {seq}…")
    video_dir = os.path.join(DATA_ROOT, seq, f"frame_{seq}")

    # 1) init and reset state
    inference_state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False
    )
    predictor.reset_state(inference_state)

    # 2) read GT box (comma‑sep)
    with open(os.path.join(GT_ROOT, f"{seq}.txt")) as f:
        x0, y0, w, h = map(float, f.readline().strip().split(","))
    # convert to [x0,y0,x1,y1]
    init_box = np.array([x0, y0, x0 + w, y0 + h], dtype=np.float32)

    # 3) initial prompt on frame 0
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=0,
        box=init_box,
    )

    # 4) propagate through video → collect boolean masks
    video_masks = {}   # frame_idx -> 2D bool mask
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        # take the first (and only) object
        mask_np = (mask_logits[0] > 0).cpu().numpy().squeeze()
        video_masks[frame_idx] = mask_np

    # 5) compute and save bboxes
    out_path = os.path.join(OUT_ROOT, "result", f"tracker_{seq}.txt")
    with open(out_path, "w") as fout:
        # iterate in frame order
        for i in range(len(video_masks)):
            x, y, w, h = mask_to_xywh(video_masks[i])
            fout.write(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")

print("\nAll done—results in", OUT_ROOT)
