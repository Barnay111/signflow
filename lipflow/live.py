"""Live sign → text: the camera stays on and every recognised sign is added to a running transcript.

    uv run lipflow live                          # built-in camera, Signflow model if trained
    uv run lipflow live --camera 1               # another camera
    uv run lipflow live --camera news.mp4 --crop br   # a video file, cropped to the interpreter

    uv run lipflow live --record                 # also record: video + transcript + subtitles

Keys in the window: R start/stop recording · Space pause/resume · C clear · S save transcript · Q / Esc quit.
A recording is saved to --out-dir (default ./recordings) as signflow-<time>.mp4 with .txt (the text and
every sign with its time in the video) and .srt subtitles, so the text appears in the player at each sign.

The capture loop keeps frames at SAMPLE_FPS in a ring buffer; a worker thread classifies the newest
WINDOW frames whenever STRIDE new ones have arrived (or as soon as it is free, if the model is slower
than that), and StreamMerger turns window labels into glosses exactly as SignReader.merge does.
"""
from __future__ import annotations

import collections
import os
import sys
import threading
import time

import cv2
import numpy as np

from .sign import DEFAULT_MODEL, MODELS, MODELS_DIR, NONE, SAMPLE_FPS, WINDOW, SignReader, letterbox, parse_crop

FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
VIEW_W, VIEW_H = 960, 600


def _font(size: int):
    from PIL import ImageFont
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw, text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class Live:
    def __init__(self, model: str, threshold: "float | None", crop_spec: "str | None"):
        self.reader = SignReader(model)
        self.reader.warmup()
        self.threshold = threshold
        self.crop_spec = crop_spec
        self.crop = None
        self.buf: collections.deque = collections.deque(maxlen=WINDOW)
        self.buf_t: collections.deque = collections.deque(maxlen=WINDOW)
        self.n_frames = 0
        self.cond = threading.Condition()
        self.running = True
        self.paused = False
        self.glosses: list[tuple[float, str, float]] = []  # (time, gloss, prob) — on screen; C clears
        self.history: list[tuple[float, str, float]] = []  # everything, for recordings
        self.merger = self.reader.stream()
        self.top: list[tuple[str, float]] = []
        self.latency = 0.0
        self.windows = 0

    # -- worker ----------------------------------------------------------------------
    def worker(self):
        last = 0
        while True:
            with self.cond:
                self.cond.wait_for(lambda: not self.running or (
                    not self.paused and len(self.buf) == WINDOW and self.n_frames - last >= self.reader.stride))
                if not self.running:
                    return
                frames, ts, last = list(self.buf), list(self.buf_t), self.n_frames
            t0 = time.time()
            p = self.reader._probs(frames)
            self.latency = time.time() - t0
            self.windows += 1
            order = np.argsort(p)[::-1][:3]
            self.top = [(self.reader.classes[i], float(p[i])) for i in order]
            k = int(order[0])
            g = self.merger.push(self.reader.label(k, float(p[k]), self.threshold))
            if g is not None:
                mid = ts[len(ts) // 2]
                self.glosses.append((mid, g, float(p[k])))
                self.history.append((mid, g, float(p[k])))
                print(f"  {mid:7.1f}s  {g}  {p[k]:.0%}", flush=True)

    def text(self) -> str:
        return SignReader.to_text([(g, p) for _, g, p in self.glosses])

    def push_frame(self, bgr: np.ndarray, t: float):
        if self.crop_spec and self.crop is None:
            self.crop = parse_crop(self.crop_spec, bgr.shape[1], bgr.shape[0])
        if self.crop:
            x, y, w, h = self.crop
            bgr = bgr[y:y + h, x:x + w]
        with self.cond:
            self.buf.append(letterbox(bgr))
            self.buf_t.append(t)
            self.n_frames += 1
            self.cond.notify()

    # -- drawing ---------------------------------------------------------------------
    def render(self, bgr: np.ndarray, mirror: bool, fonts) -> np.ndarray:
        from PIL import Image, ImageDraw
        big, small = fonts
        h, w = bgr.shape[:2]
        s = min(VIEW_W / w, VIEW_H / h)  # fit landscape and portrait video alike
        view = cv2.resize(bgr, (int(w * s), int(h * s)))
        if self.crop:  # show what the model sees
            x, y, cw, ch = (int(v * s) for v in self.crop)
            cv2.rectangle(view, (x, y), (x + cw, y + ch), (80, 220, 120), 2)
        if mirror:
            view = cv2.flip(view, 1)
        panel_h = 150
        canvas = np.zeros((view.shape[0] + panel_h, VIEW_W, 3), np.uint8)
        x0 = (VIEW_W - view.shape[1]) // 2
        canvas[:view.shape[0], x0:x0 + view.shape[1]] = view
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        # status + current guesses
        status = (f"{self.reader.name} · {'ПАУЗА' if self.paused else 'распознаю'} · "
                  f"{self.latency * 1000:.0f} мс/окно · R запись  Space пауза  C очистить  S сохранить  Q выход")
        d.rectangle([0, 0, VIEW_W, 30], fill=(0, 0, 0))
        d.text((10, 6), status, font=small, fill=(200, 200, 200))
        thr = self.threshold or self.reader.threshold
        for i, (g, p) in enumerate(self.top):
            y = 40 + i * 26
            hit = i == 0 and g != NONE and p >= thr  # this window counts as a sign
            d.rectangle([10, y, 10 + int(220 * p), y + 20], fill=(40, 110, 60) if hit else (60, 60, 60))
            d.text((16, y + 1), f"{'— нет жеста' if g == NONE else g.split(' / ')[0]}  {p:.0%}",
                   font=small, fill=(150, 255, 170) if hit else (230, 230, 230))
        # transcript
        lines = _wrap(d, self.text() or "…", big, VIEW_W - 30)[-3:]
        y0 = view.shape[0] + 12
        for i, line in enumerate(lines):
            d.text((15, y0 + i * 44), line, font=big, fill=(255, 255, 255))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text() + "\n\n")
            for t, g, p in self.glosses:
                f.write(f"{t:8.1f}\t{g}\t{p:.2f}\n")
        print(f"saved {path}")


def srt(glosses: list[tuple[float, str, float]], hold: float = 4.0, words: int = 6) -> str:
    """Subtitles: at each sign, the last few recognised words (letters joined), shown until the next
    sign or for `hold` seconds."""
    def ts(t):
        ms = int(round(max(t, 0) * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    cues = []
    for i, (t, _, _) in enumerate(glosses):
        end = min(glosses[i + 1][0] if i + 1 < len(glosses) else t + hold, t + hold)
        text = " ".join(SignReader.to_text([(g, p) for _, g, p in glosses[:i + 1]]).split()[-words:])
        cues.append(f"{len(cues) + 1}\n{ts(t)} --> {ts(max(end, t + 0.5))}\n{text}\n")
    return "\n".join(cues)


class Recorder:
    """Writes the camera to an mp4 at a constant FPS (frames repeated or dropped to follow the clock,
    so the video stays in sync with sign times) and, at the end, the signs made while recording."""
    FPS = 30.0

    def __init__(self, out_dir: str, size: tuple[int, int], t0: float):
        os.makedirs(out_dir, exist_ok=True)
        self.base = os.path.join(out_dir, time.strftime("signflow-%Y%m%d-%H%M%S"))
        self.t0, self.written = t0, 0
        for codec in ("avc1", "mp4v"):  # H.264 where OpenCV has it, else MPEG-4
            self.writer = cv2.VideoWriter(self.base + ".mp4", cv2.VideoWriter_fourcc(*codec), self.FPS, size)
            if self.writer.isOpened():
                break
        if not self.writer.isOpened():
            raise RuntimeError(f"can't write {self.base}.mp4")
        print(f"● recording → {self.base}.mp4", flush=True)

    def write(self, frame: np.ndarray, t: float):
        due = int((t - self.t0) * self.FPS) + 1
        while self.written < due:
            self.writer.write(frame)
            self.written += 1

    @property
    def seconds(self) -> float:
        return self.written / self.FPS

    def finish(self, history: list[tuple[float, str, float]]) -> list[str]:
        self.writer.release()
        end = self.t0 + self.seconds
        signs = [(t - self.t0, g, p) for t, g, p in history if self.t0 - 1.0 <= t <= end]
        text = SignReader.to_text([(g, p) for _, g, p in signs])
        with open(self.base + ".txt", "w", encoding="utf-8") as f:
            f.write((text or "(ничего не распознано)") + "\n\n")
            for t, g, p in signs:
                f.write(f"{max(t, 0):8.1f}s\t{g}\t{p:.0%}\n")
        with open(self.base + ".srt", "w", encoding="utf-8") as f:
            f.write(srt(signs))
        files = [self.base + e for e in (".mp4", ".txt", ".srt")]
        print(f"■ saved {self.seconds:.0f}s, {len(signs)} signs: {', '.join(files)}", flush=True)
        if text:
            print(f"  {text}", flush=True)
        return files


def run(camera: str = "0", model: "str | None" = None, threshold: "float | None" = None,
        crop: "str | None" = None, out: "str | None" = None, record: bool = False,
        out_dir: str = "recordings"):
    if model is None:
        model = "signflow" if os.path.exists(os.path.join(MODELS_DIR, "signflow_head.pt")) else DEFAULT_MODEL
    is_file = not str(camera).isdigit()
    if not is_file:
        os.environ.pop("OPENCV_AVFOUNDATION_SKIP_AUTH", None)  # let OpenCV ask for camera access (main thread)
    print(f"loading {model}…", flush=True)
    live = Live(model, threshold, crop)
    cap = cv2.VideoCapture(camera if is_file else int(camera))
    if not cap.isOpened():
        raise SystemExit(f"can't open camera {camera}" + ("" if is_file else
                         " — allow camera access for your terminal in System Settings → Privacy & Security → Camera"))
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    file_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else None
    th = threading.Thread(target=live.worker, daemon=True)
    th.start()
    fonts = (_font(34), _font(18))
    win = "Signflow live"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    t0, next_t, n_read = time.time(), 0.0, 0
    rec: "Recorder | None" = None
    want_rec = record
    print("window open · R record · Space pause · C clear · S save · Q quit", flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    break
                time.sleep(0.01)
                continue
            if is_file:  # play back in real time, so the model keeps up the same way as with a camera
                n_read += 1
                t = n_read / file_fps
                time.sleep(max(0.0, t0 + t - time.time()))
            else:
                t = time.time() - t0
            if t + 1e-6 >= next_t:
                while next_t <= t + 1e-6:
                    next_t += 1.0 / SAMPLE_FPS
                live.push_frame(frame, t)
            if want_rec and rec is None:
                rec = Recorder(out_dir, (frame.shape[1], frame.shape[0]), t)
            if rec is not None:
                rec.write(frame, t)
            view = live.render(frame, mirror=not is_file, fonts=fonts)
            if rec is not None:  # REC badge, top right
                cv2.circle(view, (VIEW_W - 110, 15), 7, (0, 0, 255), -1)
                cv2.putText(view, f"REC {int(rec.seconds) // 60}:{int(rec.seconds) % 60:02d}", (VIEW_W - 98, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                want_rec = rec is None
                if rec is not None:
                    rec.finish(live.history)
                    rec = None
            elif key == ord(" "):
                live.paused = not live.paused
                with live.cond:
                    live.cond.notify()
            elif key == ord("c"):
                live.glosses.clear()
            elif key == ord("s"):
                live.save(out or time.strftime("signflow-%Y%m%d-%H%M%S.txt"))
    finally:
        if rec is not None:
            time.sleep(min(live.latency + 0.2, 2.0))  # let the last window finish
            rec.finish(live.history)
        with live.cond:
            live.running = False
            live.cond.notify()
        cap.release()
        cv2.destroyAllWindows()
        th.join(timeout=3)
    print("\n" + (live.text() or "(ничего не распознано)"))
    if out:
        live.save(out)
    if live.windows:
        print(f"{live.windows} windows, last {live.latency * 1000:.0f} ms/window", file=sys.stderr)
