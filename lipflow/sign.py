"""Russian Sign Language words → text, with one of two isolated-sign models.

logos (default) — SberDevices' Logos model (EMNLP 2025, https://github.com/ai-forever/logos): MViTv2-small
    trained on Logos (~200k videos, 381 signers) + AUTSL + WLASL. 2,863 RSL glosses grouped into
    2,004 classes of visually similar signs (one class can mean "адрес / край / улица"); 97.9% top-1
    on the Logos test set. Class 0 is "no sign". Weights: CC BY 4.0 per the paper.
slovo — the earlier Slovo MViTv2-small-32-2 (https://github.com/hukenovs/slovo): 1,000 glosses + 33
    fingerspelled letters, 64% top-1. Licence: a variant of CC BY-SA 4.0, see that repo.
bukva — fingerspelling only: SberDevices' Bukva MobileNetV2-TSM (https://github.com/ai-forever/bukva),
    33 letters of the RSL manual alphabet, 83.6%, 8 frames (~0.5 s) per window, ONNX.

logos and slovo take 32 frames sampled at 15 fps (every 2nd frame of 30 fps video, ~2.1 s). A recording is cut
into overlapping windows; each window gives one class, and repeats are merged into a phrase.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import warnings

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "..", "models", "sign")
DATA = os.path.join(HERE, "data")

FRAME = 300          # frames are kept letterboxed to FRAME x FRAME; each model crops/resizes from that
SIZE = 224
WINDOW = 32          # frames per model input
SAMPLE_FPS = 15.0    # the camera keeps frames at this rate while recording
STRIDE = 8           # a new window every ~0.5 s
NONE = "---"

MODELS = {
    "logos": dict(
        file="logos.pth",
        url="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru:443/datasets/logos/logos_autsl_wlasl_model.pth",
        mean=[140.99762122, 129.92701646, 125.25081198], std=[62.07248248, 62.94645644, 61.42221137],
        threshold=0.5,
    ),
    "slovo": dict(
        file="mvit32-2.pt",
        url="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/pt/mvit32-2.pt",
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        threshold=0.35,
    ),
    "signflow": dict(  # Logos backbone + a head trained here on Slovo + Bukva (scripts/train_head.py)
        file="signflow_head.pt", url=None,
        mean=[140.99762122, 129.92701646, 125.25081198], std=[62.07248248, 62.94645644, 61.42221137],
        threshold=0.5,
    ),
    "bukva": dict(
        file="bukva_mobilenetv2_tsm.onnx",
        url="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/models/MobileNetV2_TSM.onnx",
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        threshold=0.8, window=8, stride=4, confirm=2,  # a letter counts once seen in 2 windows in a row
    ),
}
BUKVA_CLASSES = [NONE] + list("ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
DEFAULT_MODEL = "logos"


CORNERS = {  # x, y, w, h as fractions of the frame
    "br": (0.6, 0.4, 0.4, 0.6), "bl": (0.0, 0.4, 0.4, 0.6),
    "tr": (0.6, 0.0, 0.4, 0.6), "tl": (0.0, 0.0, 0.4, 0.6),
    "right": (0.5, 0.0, 0.5, 1.0), "left": (0.0, 0.0, 0.5, 1.0),
}


def parse_crop(spec: "str | None", w: int, h: int):
    """--crop: pixels "x,y,w,h", fractions "0.6,0.4,0.4,0.6", or a corner shortcut (CORNERS)."""
    if not spec:
        return None
    if spec in CORNERS:
        vals = CORNERS[spec]
    else:
        vals = tuple(float(v) for v in spec.split(","))
    if all(v <= 1.0 for v in vals):
        vals = (vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h)
    x, y, cw, ch = (int(round(v)) for v in vals)
    return max(x, 0), max(y, 0), min(cw, w - x), min(ch, h - y)


def letterbox(bgr: np.ndarray, size: int = FRAME) -> np.ndarray:
    """Resize keeping aspect ratio, pad with grey (114) to size x size, return RGB uint8."""
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ensure_model(name: str) -> str:
    m = MODELS[name]
    path = os.path.join(MODELS_DIR, m["file"])
    if not os.path.exists(path):
        if not m["url"]:
            raise FileNotFoundError(f"{path} is made by scripts/train_head.py; train it first")
        os.makedirs(MODELS_DIR, exist_ok=True)
        subprocess.run(["curl", "-fsSL", "-o", path + ".part", m["url"]], check=True)
        os.replace(path + ".part", path)
    return path


def window_starts(n: int, final: bool, window: int = WINDOW, stride: int = STRIDE) -> list[int]:
    """Start indices of the windows over n frames. While recording (final=False) only the regular
    grid; at the end also one window flush with the last frame, or a single padded one if short."""
    if n < window:
        return [0] if final and n else []
    starts = list(range(0, n - window + 1, stride))
    if final and starts[-1] != n - window:
        starts.append(n - window)
    return starts


def _logos_classes() -> list[str]:
    groups: dict[int, list[str]] = {}
    with open(os.path.join(DATA, "logos_class_ids.tsv"), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            groups.setdefault(int(row["VSSign_group_ID"]), []).append(row["lemma"])
    classes = [NONE] + [""] * max(groups)
    for k, words in groups.items():
        classes[k] = " / ".join(words)
    return classes


class _Logos:
    """Vendored MMAction2 MViT backbone + the LOGOS classification head (cls token → linear)."""

    def __init__(self, path: str):
        import torch
        from .mvit_logos import MViT
        sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        self.backbone = MViT(arch="small", drop_path_rate=0.0, dim_mul_in_attention=False).eval()
        self.backbone.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("backbone.")})
        w, b = sd["cls_head.heads.LOGOS.fc_cls.weight"], sd["cls_head.heads.LOGOS.fc_cls.bias"]
        self.fc = torch.nn.Linear(w.shape[1], w.shape[0]).eval()
        self.fc.load_state_dict({"weight": w, "bias": b})

    def __call__(self, x):  # x: N,C,T,H,W
        _, cls_token = self.backbone(x)[-1]
        return self.fc(cls_token).softmax(-1)


class SignReader:
    def __init__(self, model: str = DEFAULT_MODEL):
        import torch
        warnings.filterwarnings("ignore", message=".*torch.jit.load.*")
        self.torch = torch
        self.name = model if model in MODELS else DEFAULT_MODEL
        cfg = MODELS[self.name]
        self.mean = np.array(cfg["mean"], np.float32)
        self.std = np.array(cfg["std"], np.float32)
        self.threshold = cfg["threshold"]
        self.window_len = cfg.get("window", WINDOW)
        self.stride = cfg.get("stride", STRIDE)
        self.confirm = cfg.get("confirm", 1)
        path = ensure_model(self.name)
        if self.name == "bukva":
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
            self.model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.classes = BUKVA_CLASSES
        elif self.name in ("logos", "signflow"):
            self.model = _Logos(ensure_model("logos"))
            self.classes = _logos_classes()
            if self.name == "signflow":
                head = torch.load(path, map_location="cpu", weights_only=True)
                self.model.fc = torch.nn.Linear(head["weight"].shape[1], head["weight"].shape[0]).eval()
                self.model.fc.load_state_dict(head)
                with open(os.path.join(DATA, "signflow_classes.json"), encoding="utf-8") as f:
                    self.classes = json.load(f)
        else:
            self.model = torch.jit.load(path, map_location="cpu").eval()
            with open(os.path.join(DATA, "sign_classes.json"), encoding="utf-8") as f:
                self.classes = json.load(f)
        self.none = self.classes.index(NONE)
        self.lock = threading.Lock()

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """FRAME x FRAME letterboxed RGB → the model's 224 x 224 input."""
        if self.name in ("logos", "signflow"):  # as in Logos' test pipeline: letterbox to 300, centre crop 224
            o = (FRAME - SIZE) // 2
            return frame[o:o + SIZE, o:o + SIZE]
        return cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)  # Slovo: letterbox to 224

    def warmup(self):
        self._probs([np.zeros((FRAME, FRAME, 3), np.uint8)] * self.window_len)

    def _probs(self, window: list[np.ndarray]) -> np.ndarray:
        x = (np.stack([self.prepare(f) for f in window]).astype(np.float32) - self.mean) / self.std  # T,H,W,C
        if self.name == "bukva":  # 1,T,C,H,W
            x = np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2))[None])
            with self.lock:
                out = self.model.run(None, {self.model.get_inputs()[0].name: x})[0][0]
            return out
        x = np.ascontiguousarray(np.transpose(x, (3, 0, 1, 2))[None])  # 1,C,T,H,W
        with self.lock, self.torch.inference_mode():
            t = self.torch.from_numpy(x)
            out = self.model(t if self.name in ("logos", "signflow") else t[None])[0].numpy()
        if out.min() < 0 or abs(out.sum() - 1) > 1e-3:  # logits → probabilities
            e = np.exp(out - out.max())
            out = e / e.sum()
        return out

    def window(self, frames: list[np.ndarray], start: int) -> tuple[int, float]:
        w = frames[start:start + self.window_len]
        w = w + [w[-1]] * (self.window_len - len(w))  # a short clip: pad with its last frame
        p = self._probs(w)
        k = int(p.argmax())
        return k, float(p[k])

    def label(self, k: int, p: float, threshold: "float | None" = None) -> str:
        return self.classes[k] if k != self.none and p >= (threshold or self.threshold) else NONE

    def read(self, frames: list[np.ndarray], cache: "dict | None" = None, final: bool = True
             ) -> list[tuple[str, float]]:
        """Glosses with their confidence, in order, repeats merged. `cache` (start → result) lets
        windows computed while recording be reused at the end."""
        cache = {} if cache is None else cache
        n = len(frames)
        labels = []
        for s in window_starts(n, final, self.window_len, self.stride):
            key = (s, min(n - s, self.window_len))
            if key not in cache:
                cache[key] = self.window(frames, s)
            k, p = cache[key]
            labels.append((self.label(k, p), p))
        return [(g, p) for _, g, p in self.merge(labels)]

    def merge(self, labels: list[tuple[str, float]]) -> list[tuple[int, str, float]]:
        """Per-window labels → (first window index, gloss, best confidence): a gloss needs `confirm`
        windows in a row (filters the hand in transit), and a run of the same gloss counts once."""
        out: list[tuple[int, str, float]] = []
        i = 0
        while i < len(labels):
            g, j, best = labels[i][0], i, 0.0
            while j < len(labels) and labels[j][0] == g:
                best = max(best, labels[j][1])
                j += 1
            if g != NONE and j - i >= self.confirm and not (out and out[-1][1] == g and
                                                            all(x[0] == NONE for x in labels[out[-1][0]:i])):
                out.append((i, g, best))
            i = j
        return out

    def stream(self) -> "StreamMerger":
        return StreamMerger(self.confirm)

    @staticmethod
    def to_text(glosses: list[tuple[str, float]]) -> str:
        """Fingerspelled letters join into a word; whole-word signs are separated by spaces. A Logos
        class of look-alike signs ("адрес / край / улица") is typed as its first word."""
        parts, word = [], ""
        for g, _ in glosses:
            g = g.split(" / ")[0]
            if len(g) == 1 and g.isalpha():
                word += g
            else:
                if word:
                    parts.append(word.capitalize())
                    word = ""
                parts.append(g)
        if word:
            parts.append(word.capitalize())
        return " ".join(parts)


class StreamMerger:
    """SignReader.merge for an endless stream: feed one window label at a time and get a gloss back
    as soon as it is confirmed (`confirm` windows in a row; a run of the same gloss counts once)."""

    def __init__(self, confirm: int = 1):
        self.confirm = confirm
        self.label, self.run = None, 0

    def push(self, label: str) -> "str | None":
        self.run = self.run + 1 if label == self.label else 1
        self.label = label
        return label if label != NONE and self.run == self.confirm else None
