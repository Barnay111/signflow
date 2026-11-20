"""Train Signflow's classifier head on frozen Logos features (see extract_features.py).

    uv run python scripts/train_head.py            # uses whatever of datasets/features/{slovo,bukva} exists

Classes = the 2,005 Logos classes (0 = no sign) + Slovo words Logos doesn't have + the 33 letters.
Rows of Logos classes that get training data start from Logos' own weights and stay close to them;
rows of Logos classes with no data are frozen, so the model doesn't forget them; new rows are learnt.
Selection is on a validation split of the training signers; the score reported is on the datasets'
own test splits (signers never seen in training), next to the untouched Logos head as a baseline.

Writes models/sign/signflow_head.pt and lipflow/data/signflow_classes.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lipflow.sign import MODELS_DIR, NONE, _logos_classes  # noqa: E402

FEATS = os.path.join(os.environ.get("SIGNFLOW_DATASETS") or os.path.join(ROOT, "datasets"), "features")
OUT_HEAD = os.path.join(MODELS_DIR, "signflow_head.pt")
OUT_CLASSES = os.path.join(ROOT, "lipflow", "data", "signflow_classes.json")
EPOCHS, BATCH, LR, ANCHOR = 60, 256, 1e-3, 1e-2


def norm(t: str) -> str:
    return re.sub(r"[^\w\s-]", "", t.lower().replace("ё", "е")).strip()


def load(name):
    parts = [np.load(p, allow_pickle=False) for p in sorted(glob.glob(os.path.join(FEATS, name, "*.npz")))]
    if not parts:
        return None
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return dict(feat=cat("feat"), text=cat("text"), user=cat("user"), train=cat("train"),
                src=np.array([name] * sum(len(p["ids"]) for p in parts)))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    data = [d for d in (load("slovo"), load("bukva")) if d is not None]
    if not data:
        raise SystemExit("no features yet: run scripts/extract_features.py first")
    D = {k: np.concatenate([d[k] for d in data]) for k in data[0]}

    logos = _logos_classes()
    lemma_to_class = {}
    for k, group in enumerate(logos):
        if k:
            for w in group.split(" / "):
                lemma_to_class.setdefault(norm(w), k)
    classes = list(logos)
    extra: dict[str, int] = {}

    def label(text: str) -> int:
        t = text.strip()
        if t in ("---", "no_event", "no event") or not t:
            return 0
        if len(t) == 1 and t.isalpha():  # a fingerspelled letter (Slovo and Bukva share these)
            key = t.upper()
        else:
            k = lemma_to_class.get(norm(t))
            if k is not None:
                return k
            key = t
        if key not in extra:
            extra[key] = len(classes)
            classes.append(key)
        return extra[key]

    y = np.array([label(t) for t in D["text"]])
    X = torch.from_numpy(D["feat"])
    Y = torch.from_numpy(y)
    n_old = len(logos)
    print(f"{len(y)} clips; {len(classes)} classes = {n_old} Logos + {len(classes) - n_old} new "
          f"({sum(1 for c in classes[n_old:] if len(c) == 1)} letters)")
    mapped = sorted({int(c) for c in y if 0 < c < n_old})
    print(f"training data reaches {len(mapped)} Logos classes; {n_old - 1 - len(mapped)} stay frozen")

    # splits: dataset test sets for the final score; 10% of training signers for model selection
    test = ~D["train"].astype(bool)
    users, counts = np.unique(D["user"][~test], return_counts=True)
    # hold out whole signers until they cover ~10% of training *clips* (signers differ a lot in
    # how many clips they recorded, so 10% of signers can be half the data)
    budget, val_users, taken = 0.1 * (~test).sum(), set(), 0
    for i in np.random.permutation(len(users)):
        if taken + counts[i] <= budget:
            val_users.add(users[i])
            taken += counts[i]
    if len(users) >= 10 and val_users:
        val = ~test & np.isin(D["user"], list(val_users))
    else:  # too few signers to hold some out: a random 10% of training clips instead
        val = ~test & (np.random.rand(len(y)) < 0.1)
        val_users = set()
    tr = ~test & ~val
    print(f"train {tr.sum()}, val {val.sum()} ({len(val_users)} signers), test {test.sum()}")

    sd = torch.load(os.path.join(MODELS_DIR, "logos.pth"), map_location="cpu", weights_only=True)["state_dict"]
    W0, b0 = sd["cls_head.heads.LOGOS.fc_cls.weight"], sd["cls_head.heads.LOGOS.fc_cls.bias"]
    head = torch.nn.Linear(X.shape[1], len(classes))
    with torch.no_grad():
        head.weight.normal_(0, 0.01)
        head.bias.fill_(float(b0.mean()))
        head.weight[:n_old] = W0
        head.bias[:n_old] = b0
    trainable = torch.zeros(len(classes), dtype=torch.bool)
    trainable[n_old:] = True
    trainable[mapped] = True
    trainable[0] = True
    grad_mask = trainable.float()[:, None]
    head.weight.register_hook(lambda g: g * grad_mask)
    head.bias.register_hook(lambda g: g * grad_mask[:, 0])

    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    tr_idx = torch.from_numpy(np.where(tr)[0])

    def acc(mask, w=None, b=None, restrict=None):
        idx = np.where(mask)[0]
        if restrict is not None:
            idx = idx[restrict(y[idx])]
        if not len(idx):
            return float("nan"), 0
        with torch.no_grad():
            logits = X[idx] @ (head.weight if w is None else w).T + (head.bias if b is None else b)
            pred = logits.argmax(1).numpy()
        return float((pred == y[idx]).mean()), len(idx)

    best, best_state, loss = -1.0, None, torch.tensor(float("nan"))
    for ep in range(EPOCHS):
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for s in range(0, len(perm), BATCH):
            i = perm[s:s + BATCH]
            loss = lossf(head(X[i]), Y[i])
            loss = loss + ANCHOR * ((head.weight[:n_old] - W0) ** 2).sum() / n_old
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        v, _ = acc(val)
        if v > best:
            best, best_state = v, {k: t.clone() for k, t in head.state_dict().items()}
        if ep % 10 == 9 or ep == EPOCHS - 1:
            print(f"  epoch {ep + 1}: loss {loss.item():.3f}, val {v:.1%}")
    head.load_state_dict(best_state)

    print("\nTest (signers never seen in training):")
    for src in np.unique(D["src"]):
        m = test & (D["src"] == src)
        a, n = acc(m)
        print(f"  {src}: new head {a:.1%} on {n} clips")
        in_logos = lambda yy: (yy > 0) & (yy < n_old)
        a_new, n_in = acc(m, restrict=in_logos)
        a_old, _ = acc(m, w=W0, b=b0, restrict=in_logos)
        if n_in:
            print(f"    words Logos already had ({n_in} clips): Logos head {a_old:.1%} → new head {a_new:.1%}")
        a_extra, n_extra = acc(m, restrict=lambda yy: yy >= n_old)
        if n_extra:
            print(f"    new classes ({n_extra} clips): {a_extra:.1%} (Logos head: 0%, it can't output them)")

    torch.save({"weight": head.weight.detach().clone(), "bias": head.bias.detach().clone()}, OUT_HEAD)
    with open(OUT_CLASSES, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)
    print(f"\nsaved {OUT_HEAD} and {OUT_CLASSES}")


if __name__ == "__main__":
    main()
