"""Run the sign model over a video file and print the recognised RSL words with timestamps.

    uv run python scripts/sign_video.py VIDEO [--crop x,y,w,h | --crop br|bl|tr|tl|right|left] [--preview out.jpg]

An interpreter is usually a small inset in a corner of a broadcast, while the model was trained on a
person filling the frame, so crop to the interpreter. --crop takes pixels (x,y,w,h), fractions of the
frame (0.65,0.5,0.35,0.5) or a corner shortcut. --preview saves the cropped first frame to check it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lipflow.sign import DEFAULT_MODEL, MODELS, NONE, SAMPLE_FPS, WINDOW, SignReader, letterbox, parse_crop, window_starts  # noqa: E402

def ts(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--crop")
    ap.add_argument("--preview", help="save the cropped first frame here and exit")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=None, help="seconds")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--threshold", type=float, default=None, help="default: the model's own")
    ap.add_argument("--slow", type=float, default=1.0,
                    help="play the video this many times slower for the model (2 = half speed)")
    ap.add_argument("--frames", help="save a grid of cropped frames here to check the crop, and exit")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"can't open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    crop = parse_crop(a.crop, w, h)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)

    frames, times = [], []
    next_t = a.start
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if a.end is not None and t > a.end:
            break
        if t + 1e-6 < next_t:
            continue
        reps = 0
        while next_t <= t + 1e-6:  # slowed down past the source rate: a frame is used more than once
            next_t += 1.0 / (SAMPLE_FPS * a.slow)
            reps += 1
        if crop:
            x, y, cw, ch = crop
            f = f[y:y + ch, x:x + cw]
        if a.preview:
            cv2.imwrite(a.preview, f)
            print(f"saved {a.preview} ({f.shape[1]}x{f.shape[0]} of {w}x{h})")
            return
        img = letterbox(f)
        frames.extend([img] * reps)
        times.extend([t] * reps)
    print(f"{a.video}: {w}x{h}, {fps:.0f} fps, {total:.0f}s; using {len(frames)} frames "
          f"({ts(times[0]) if times else '-'}–{ts(times[-1]) if times else '-'})"
          + (f", crop {crop}" if crop else ""))
    if not frames:
        return
    if a.frames:
        import numpy as np
        pick = [frames[i] for i in np.linspace(0, len(frames) - 1, 8).astype(int)]
        grid = np.vstack([np.hstack(pick[:4]), np.hstack(pick[4:])])
        cv2.imwrite(a.frames, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {a.frames}")
        return

    r = SignReader(a.model)
    t0 = time.time()
    starts = window_starts(len(frames), True, r.window_len, r.stride)
    labels = []
    for i, s in enumerate(starts):
        k, p = r.window(frames, s)
        labels.append((r.label(k, p, a.threshold), p))
        if i % 20 == 19:
            print(f"  … {i + 1}/{len(starts)} windows, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    words = []
    for i, g, p in r.merge(labels):
        mid = times[min(starts[i] + r.window_len // 2, len(times) - 1)]
        words.append((mid, g, p))
        print(f"  {ts(mid)}  {g:<24} {p:.0%}", flush=True)
    print(f"\n{len(words)} words in {time.time() - t0:.0f}s (model {a.model}, slow x{a.slow}):")
    print(" ".join(g for _, g, _ in words) or "(nothing above the threshold)")


if __name__ == "__main__":
    main()
