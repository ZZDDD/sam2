import os
import time
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from zipfile import ZipFile
import re


# —— CONFIG ——
# paths (adjust if yours differ)
DATA_ROOT     = "/home/dell/sda/datasets/VideoCube/MGIT/data/test"
GT_ROOT       = "/home/dell/sda/datasets/VideoCube/sam_track_homework/groundtruth"
TEST_LIST_TXT = "/home/dell/sda/datasets/VideoCube/sam_track_homework/test_list.txt"
OUT_DIR       = "./tracker"    # will contain result/ and time/
ZIP_NAME      = "sam2_results.zip"

# choose your checkpoint & config
SAM2_CKPT  = "/home/dell/repositories/sam2/checkpoints/sam2.1_hiera_tiny.pt"
# SAM2_CFG   = "/home/dell/repositories/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_t.yaml"

# —— HELPERS ——
def mask_to_bbox(mask: np.ndarray):
    """Compute tight bbox [xmin,ymin,xmax,ymax] from a boolean mask."""
    ys, xs = np.where(mask)
    if len(xs)==0:
        return 0,0,0,0
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1, y1

def xyxy_to_xywh(x0, y0, x1, y1):
    w = x1 - x0
    h = y1 - y0
    return x0, y0, w, h

# —— MAIN ——
def main():
    os.makedirs(os.path.join(OUT_DIR,"result"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR,"time"),   exist_ok=True)

    # build predictor once
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else
                          "cpu")
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)

    # read test list
    with open(TEST_LIST_TXT, "r") as f:
        seqs = [line.strip().zfill(3) for line in f if line.strip()]

    for seq in seqs:
        print(f"→ Processing sequence {seq} …")
        video_dir = os.path.join(DATA_ROOT, seq, f"frame_{seq}")
        # init state (loads all frames)
        inference_state = predictor.init_state(
            video_path=video_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            # async_loading_frames=True
        )

        # read GT, take first line
        gt_file = os.path.join(GT_ROOT, f"{seq}.txt")
        with open(gt_file, "r") as f:
            line = f.readline().strip()
        parts = re.split(r'[\s,]+', line)
        x0, y0, w, h = map(float, parts[:4])
        box = np.array([x0, y0, x0 + w, y0 + h], dtype=np.float32)
        # x0, y0, w, h = map(float, line.split())
        # box = np.array([x0, y0, x0+w, y0+h], dtype=np.float32)

        # reset (in case)
        # predictor.reset_state(inference_state)

        bboxes = []
        times  = []

        # — initial prompt on frame 0 —
        t0 = time.time()
        _, obj_ids, mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=0,
            box=box
        )
        t1 = time.time()
        times.append(t1 - t0)

        # compute bbox from mask
        mask0 = (mask_logits[0] > 0.0).cpu().numpy().squeeze()

        x0_, y0_, x1_, y1_ = mask_to_bbox(mask0)
        bboxes.append(xyxy_to_xywh(x0_,y0_,x1_,y1_))

        # — propagate through video —
        prev_time = time.time()
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            now = time.time()
            times.append(now - prev_time)
            prev_time = now

            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

            x0_, y0_, x1_, y1_ = mask_to_bbox(mask)
            bboxes.append(xyxy_to_xywh(x0_,y0_,x1_,y1_))

        # write result file
        res_path = os.path.join(OUT_DIR, "result", f"tracker_{seq}.txt")
        with open(res_path, "w") as f:
            for (x,y,w,h) in bboxes:
                f.write(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")

        # write time file
        time_path = os.path.join(OUT_DIR, "time", f"tracker_{seq}.txt")
        with open(time_path, "w") as f:
            for t in times:
                f.write(f"{t:.6f}\n")

    # finally, zip up the whole tracker/ folder
    with ZipFile(ZIP_NAME, 'w') as zipf:
        for root, _, files in os.walk(OUT_DIR):
            for fn in files:
                full = os.path.join(root, fn)
                arc  = os.path.relpath(full, start=os.path.dirname(OUT_DIR))
                zipf.write(full, arc)

    print(f"\n✅ Done! Submission archive: {ZIP_NAME}")

if __name__ == "__main__":
      main()
