"""Run the frozen Logos backbone over a dataset once and save one 768-d feature per clip.

    uv run python scripts/extract_features.py slovo   # datasets/slovo  → datasets/features/slovo/*.npz
    uv run python scripts/extract_features.py bukva   # datasets/bukva  → datasets/features/bukva/*.npz

Each clip gives 32 frames at every 2nd frame, centred on the annotated sign (begin..end), prepared
exactly as the app does at run time (letterbox to 300, centre crop 224). Work is saved in shards of
SHARD clips, so an interrupted run continues where it stopped. Two worker processes share the CPU.
"""
from __future__ import annotations

import csv
import glob
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
DATASETS = os.environ.get("SIGNFLOW_DATASETS") or os.path.join(ROOT, "datasets")
SHARD = 200
WORKERS = 2
THREADS = 5


def annotations(name: str) -> list[dict]:
    base = os.path.join(DATASETS, name)
    path = (glob.glob(os.path.join(base, "**", "annotations.csv"), recursive=True)
            + glob.glob(os.path.join(base, "**", "annotations.tsv"), recursive=True))[0]
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t" if path.endswith(".tsv") else ","))
    if name == "bukva":  # prefer the trimmed clips (the sign only); begin/end refer to the originals
        folder = glob.glob(os.path.join(base, "**", "trimmed"), recursive=True)
        trimmed = folder[0] if folder else None
    videos = {}
    for p in glob.glob(os.path.join(base, "**", "*.*"), recursive=True):
        if p.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv")):
            if name == "bukva" and trimmed and not p.startswith(trimmed):
                continue
            videos[os.path.splitext(os.path.basename(p))[0]] = p
    out = []
    for r in rows:
        v = videos.get(r["attachment_id"])
        if v is None:
            continue
        trimmed_clip = name == "bukva" and trimmed is not None
        out.append(dict(id=r["attachment_id"], path=v, text=r["text"], user=r["user_id"],
                        train=str(r["train"]).strip().lower() == "true",
                        begin=None if trimmed_clip else int(float(r["begin"])),
                        end=None if trimmed_clip else int(float(r["end"]))))
    return out


def clip_frames(path: str, begin, end, n: int = 32, step: int = 2) -> "list[np.ndarray] | None":
    from lipflow.sign import letterbox
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return None
    b = 0 if begin is None else max(0, min(begin, total - 1))
    e = total - 1 if end is None else max(b, min(end, total - 1))
    mid = (b + e) // 2
    idx = [min(max(mid - n + i * step, 0), total - 1) for i in range(n)]  # centred, clamped (repeat edges)
    frames, cur, last = {}, -1, None
    want = sorted(set(idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
    cur = want[0] - 1
    for w in want:
        while cur < w:
            ok, f = cap.read()
            cur += 1
            if not ok:
                break
            last = f
        if last is None:
            return None
        frames[w] = letterbox(last)
    return [frames[i] for i in idx]


_reader = None


def _init():
    global _reader
    import torch
    torch.set_num_threads(THREADS)
    from lipflow.sign import SignReader
    _reader = SignReader("logos")


def _feature(item):
    import torch
    fr = clip_frames(item["path"], item["begin"], item["end"])
    if fr is None:
        return item["id"], None
    r = _reader
    x = (np.stack([r.prepare(f) for f in fr]).astype(np.float32) - r.mean) / r.std
    x = torch.from_numpy(np.ascontiguousarray(np.transpose(x, (3, 0, 1, 2))[None]))
    with torch.inference_mode():
        _, cls_token = r.model.backbone(x)[-1]
    return item["id"], cls_token[0].numpy().astype(np.float32)


def main():
    name = sys.argv[1]
    items = annotations(name)
    out_dir = os.path.join(DATASETS, "features", name)
    os.makedirs(out_dir, exist_ok=True)
    done = set()
    for p in glob.glob(os.path.join(out_dir, "shard_*.npz")):
        done.update(np.load(p, allow_pickle=False)["ids"].tolist())
    todo = [it for it in items if it["id"] not in done]
    print(f"{name}: {len(items)} clips, {len(done)} done, {len(todo)} to go", flush=True)
    t0, n_ok = time.time(), 0
    with Pool(WORKERS, initializer=_init) as pool:
        for s in range(0, len(todo), SHARD):
            batch = todo[s:s + SHARD]
            res = pool.map(_feature, batch, chunksize=4)
            keep = [(it, f) for it, (_, f) in zip(batch, res) if f is not None]
            if keep:
                np.savez(os.path.join(out_dir, f"shard_{int(time.time() * 1000)}.npz"),
                         ids=np.array([it["id"] for it, _ in keep]),
                         text=np.array([it["text"] for it, _ in keep]),
                         user=np.array([it["user"] for it, _ in keep]),
                         train=np.array([it["train"] for it, _ in keep]),
                         feat=np.stack([f for _, f in keep]))
            n_ok += len(keep)
            el = time.time() - t0
            left = el / max(s + len(batch), 1) * (len(todo) - s - len(batch))
            print(f"  {s + len(batch)}/{len(todo)}  ({len(batch) - len(keep)} unreadable)  "
                  f"{el / 60:.0f} min, ~{left / 60:.0f} min left", flush=True)
    print(f"done: {n_ok} features in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
