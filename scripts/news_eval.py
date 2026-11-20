"""How much of what a news interpreter signs does the sign model read? Compare against the speech.

    uv run --group eval python scripts/news_eval.py signs   VIDEO --crop x,y,w,h   # → VIDEO.signs.npz
    uv run --group eval python scripts/news_eval.py speech  VIDEO                  # → VIDEO.speech.json (Whisper)
    uv run --group eval python scripts/news_eval.py compare VIDEO [--model signflow] [--threshold 0.5]

signs:   slides the model over the cropped interpreter exactly as the app does (15 fps, 32-frame windows,
         a new one every 8 frames) and stores the backbone token of every window, so any head and any
         threshold can be scored afterwards without re-running the video.
speech:  Whisper large-v3-turbo (mlx, Apple GPU) with word timestamps. The reference text.
compare: lemmatises both sides (pymorphy3) and matches each recognised sign to the speech around it.
         An interpreter runs a few seconds behind the speaker, so a sign at time t matches a spoken word
         in [t - LAG_BEFORE, t + LAG_AFTER]. Chance level: the same signs with their times shifted
         by a minute or more, so frequent words ("год", "человек") don't pass for recognition.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lipflow.sign import MODELS, MODELS_DIR, NONE, SAMPLE_FPS, STRIDE, WINDOW, SignReader, _logos_classes, letterbox, parse_crop  # noqa: E402

LAG_BEFORE, LAG_AFTER = 10.0, 3.0  # seconds: spoken word may be this much before / after the sign
FUNCTION_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "PRED", "NUMR", None}


# -- 1. signs --------------------------------------------------------------------------
def signs(video: str, crop_spec: "str | None", threads: int):
    import torch
    torch.set_num_threads(threads)
    r = SignReader("logos")  # backbone only; heads are applied in compare
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    crop = parse_crop(crop_spec, w, h)
    buf, buf_t, toks, mids = collections.deque(maxlen=WINDOW), collections.deque(maxlen=WINDOW), [], []
    n, next_t, since, t0 = 0, 0.0, 0, time.time()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = n / fps
        n += 1
        if t + 1e-6 < next_t:
            continue
        while next_t <= t + 1e-6:
            next_t += 1.0 / SAMPLE_FPS
        if crop:
            x, y, cw, ch = crop
            f = f[y:y + ch, x:x + cw]
        buf.append(letterbox(f))
        buf_t.append(t)
        since += 1
        if len(buf) == WINDOW and since >= STRIDE:
            since = 0
            x = (np.stack([r.prepare(fr) for fr in buf]).astype(np.float32) - r.mean) / r.std
            x = torch.from_numpy(np.ascontiguousarray(np.transpose(x, (3, 0, 1, 2))[None]))
            with torch.inference_mode():
                _, tok = r.model.backbone(x)[-1]
            toks.append(tok[0].numpy())
            mids.append(buf_t[WINDOW // 2])
            if len(toks) % 100 == 0:
                el = time.time() - t0
                print(f"  {t:6.0f}/{dur:.0f}s video, {len(toks)} windows, {el / 60:.1f} min", flush=True)
    out = os.path.splitext(video)[0] + ".signs.npz"
    np.savez(out, tok=np.stack(toks), t=np.array(mids), crop=np.array(crop or (0, 0, w, h)))
    print(f"saved {out}: {len(toks)} windows in {(time.time() - t0) / 60:.1f} min")


# -- 2. speech -------------------------------------------------------------------------
def speech(video: str, model: str):
    import mlx_whisper
    t0 = time.time()
    res = mlx_whisper.transcribe(video, path_or_hf_repo=model, language="ru", word_timestamps=True,
                                 condition_on_previous_text=False)
    words = [dict(w=w["word"].strip(), s=float(w["start"]), e=float(w["end"]))
             for seg in res["segments"] for w in seg.get("words", [])]
    out = os.path.splitext(video)[0] + ".speech.json"
    json.dump(dict(text=res["text"], words=words), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"saved {out}: {len(words)} words in {(time.time() - t0) / 60:.1f} min")


# -- 3. compare ------------------------------------------------------------------------
_morph = None


def lemmas(text: str) -> list[tuple[str, "str | None"]]:
    """(lemma, part of speech) per word."""
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    out = []
    for w in re.findall(r"[а-яёa-z]+", text.lower().replace("ё", "е")):
        p = _morph.parse(w)[0]
        out.append((p.normal_form.replace("ё", "е"), p.tag.POS))
    return out


def gloss_lemmas(g: str) -> set[str]:
    """'адрес / край / улица' → {адрес, край, улица}; a multi-word gloss gives its content words."""
    out = set()
    for alt in g.split(" / "):
        ls = [l for l, pos in lemmas(alt) if pos not in FUNCTION_POS] or [l for l, _ in lemmas(alt)]
        out.update(ls)
    return out


def decode(tok: np.ndarray, model: str, threshold: "float | None"):
    """Window tokens → per-window labels with the chosen head, as SignReader.label does."""
    if model == "signflow":
        import torch
        h = torch.load(os.path.join(MODELS_DIR, "signflow_head.pt"), map_location="cpu", weights_only=True)
        W, b = h["weight"].numpy(), h["bias"].numpy()
        classes = json.load(open(os.path.join(ROOT, "lipflow", "data", "signflow_classes.json"), encoding="utf-8"))
    else:
        import torch
        sd = torch.load(os.path.join(MODELS_DIR, "logos.pth"), map_location="cpu", weights_only=True)["state_dict"]
        W, b = sd["cls_head.heads.LOGOS.fc_cls.weight"].numpy(), sd["cls_head.heads.LOGOS.fc_cls.bias"].numpy()
        classes = _logos_classes()
    z = tok @ W.T + b
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    k = p.argmax(1)
    thr = threshold or MODELS[model]["threshold"]
    none = classes.index(NONE)
    labels = [(classes[i] if i != none and p[j, i] >= thr else NONE, float(p[j, i])) for j, i in enumerate(k)]
    return labels, classes


class _M:
    merge = SignReader.merge
    confirm = 1


def recognised(video: str, model: str, threshold):
    d = np.load(os.path.splitext(video)[0] + ".signs.npz")
    labels, classes = decode(d["tok"], model, threshold)
    out = []  # (time, text, prob): fingerspelled letters joined into one word, like the app types them
    word, wt, wp = "", None, 0.0
    for i, g, p in _M().merge(labels):
        t = float(d["t"][i])
        if len(g) == 1 and g.isalpha():
            word, wt, wp = word + g, wt if word else t, max(wp, p)
            continue
        if word:
            out.append((wt, word.lower(), wp, True))
            word, wp = "", 0.0
        out.append((t, g, p, False))
    if word:
        out.append((wt, word.lower(), wp, True))
    vocab = set()
    for c in classes:
        if c != NONE and not (len(c) == 1 and c.isalpha()):
            vocab |= gloss_lemmas(c)
    return out, vocab, float(d["t"][-1]) if len(d["t"]) else 0.0


def compare(video: str, model: str, threshold, report: "str | None"):
    sp = json.load(open(os.path.splitext(video)[0] + ".speech.json", encoding="utf-8"))
    spoken = []  # (time, lemma, is_content)
    for w in sp["words"]:
        for l, pos in lemmas(w["w"]):
            spoken.append(((w["s"] + w["e"]) / 2, l, pos not in FUNCTION_POS))
    sg, vocab, dur = recognised(video, model, threshold)
    s_t = np.array([s[0] for s in spoken])
    s_l = [s[1] for s in spoken]

    def hit(t, ls):
        m = (s_t >= t - LAG_BEFORE) & (s_t <= t + LAG_AFTER)
        return [(s_t[i] - t) for i in np.where(m)[0] if s_l[i] in ls]

    sign_ls = [(t, gloss_lemmas(g) if not letters else {g}, g, p, letters) for t, g, p, letters in sg]
    matches = [hit(t, ls) for t, ls, *_ in sign_ls]
    n = len(sign_ls)
    prec = sum(bool(m) for m in matches) / max(n, 1)
    # chance: the same signs, times shifted circularly by ≥ 60 s
    rng = np.random.default_rng(0)
    chance = []
    for _ in range(50):
        sh = rng.uniform(60, max(dur - 60, 61))
        chance.append(np.mean([bool(hit((t + sh) % dur, ls)) for t, ls, *_ in sign_ls]) if n else 0)
    # recall over spoken content words the model has a sign for
    content = [(t, l) for t, l, c in spoken if c]
    in_vocab = [(t, l) for t, l in content if l in vocab]
    sign_t = np.array([s[0] for s in sign_ls]) if n else np.zeros(0)
    rec_hits = 0
    for t, l in in_vocab:
        m = np.where((sign_t >= t - LAG_AFTER) & (sign_t <= t + LAG_BEFORE))[0]
        rec_hits += any(l in sign_ls[i][1] for i in m)
    lags = [x for m in matches for x in m]

    lines = [f"# {os.path.basename(video)} · model {model}, threshold {threshold or MODELS[model]['threshold']}",
             f"video {dur / 60:.1f} min · speech {len(spoken)} words ({len(content)} content words) · "
             f"recognised signs {n} ({sum(s[4] for s in sign_ls)} fingerspelled words)",
             "",
             f"precision  {prec:6.1%}  of recognised signs had a matching spoken word within "
             f"-{LAG_BEFORE:.0f}…+{LAG_AFTER:.0f} s   (chance with shifted times: {np.mean(chance):.1%})",
             f"coverage   {len(in_vocab) / max(len(content), 1):6.1%}  of spoken content words are in the model's vocabulary",
             f"recall     {rec_hits / max(len(in_vocab), 1):6.1%}  of those in-vocabulary spoken words were recognised as signs",
             f"lag        median {np.median(lags) if lags else float('nan'):+.1f} s (speech minus sign) over matches",
             "", "## matched signs"]
    for (t, ls, g, p, L), m in zip(sign_ls, matches):
        if m:
            lines.append(f"  {t:7.1f}s  {g}  ({p:.0%})")
    lines += ["", "## timeline (30 s blocks): speech | signs (✓ = also spoken nearby)"]
    for b0 in np.arange(0, dur + 30, 30):
        st = " ".join(w["w"] for w in sp["words"] if b0 <= w["s"] < b0 + 30)
        sgs = " ".join(("✓" if m else "") + g.split(" / ")[0]
                       for (t, ls, g, p, L), m in zip(sign_ls, matches) if b0 <= t < b0 + 30)
        if st or sgs:
            lines += [f"[{int(b0 // 60):02d}:{int(b0 % 60):02d}] РЕЧЬ:  {st}", f"        ЖЕСТЫ: {sgs or '—'}", ""]
    txt = "\n".join(lines)
    print("\n".join(lines[:8]))
    out = report or os.path.splitext(video)[0] + f".{model}.report.txt"
    open(out, "w", encoding="utf-8").write(txt)
    print(f"\nfull report: {out}")
    return dict(video=os.path.basename(video), model=model, signs=n, precision=prec, chance=float(np.mean(chance)),
                coverage=len(in_vocab) / max(len(content), 1), recall=rec_hits / max(len(in_vocab), 1),
                content=len(content), in_vocab=len(in_vocab), minutes=dur / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["signs", "speech", "compare"])
    ap.add_argument("video")
    ap.add_argument("--crop")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--whisper", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--model", default="signflow", choices=["signflow", "logos"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.step == "signs":
        signs(a.video, a.crop, a.threads)
    elif a.step == "speech":
        speech(a.video, a.whisper)
    else:
        print(json.dumps(compare(a.video, a.model, a.threshold, a.report), ensure_ascii=False))


if __name__ == "__main__":
    main()
