"""
STEP 1 — FRAME EXTRACTOR (fixed)
=================================
Single sequential pass through the video — no seeking.
Should complete in 1-3 minutes for a 3-minute video.

Usage:
    python step1_extract_frames.py --video "data/raw/bestcourtshots.mp4"
                                   --output frames/
                                   --count 2000
"""

import cv2
import os
import argparse
import numpy as np


def phash(frame, size=16) -> int:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size))
    bits  = (small > small.mean()).flatten()
    h     = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def extract_frames(video_path, output_dir, target_count=2000,
                   hash_thresh=8):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    # Sample every Nth frame to spread evenly across the video
    step   = max(1, total // target_count)

    print(f"📹 Video : {video_path}")
    print(f"   Frames : {total}  |  FPS: {fps:.0f}  |  "
          f"Duration: {total/fps:.1f}s")
    print(f"   Sampling every {step} frames → target {target_count} frames")
    print(f"   Output : {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    saved        = 0
    seen_hashes  = []
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth frame
        if frame_idx % step == 0:
            # Dedup check
            h    = phash(frame)
            dupe = any(hamming(h, sh) < hash_thresh for sh in seen_hashes[-200:])
            # Only check last 200 hashes for speed

            if not dupe:
                seen_hashes.append(h)
                fname = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1

                if saved % 50 == 0:
                    pct = 100 * frame_idx / total
                    print(f"  [{pct:5.1f}%] Saved {saved} frames...", end='\r')

                if saved >= target_count:
                    break

        frame_idx += 1

    cap.release()
    print(f"\n\n✅ Done! Saved {saved} frames → {output_dir}")
    print("Next: run  step2_autolabel.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default="data/raw/bestcourtshots.mp4")
    parser.add_argument("--output", default="frames/")
    parser.add_argument("--count",  type=int, default=2000)
    parser.add_argument("--dedup",  type=int, default=8)
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.count, args.dedup)