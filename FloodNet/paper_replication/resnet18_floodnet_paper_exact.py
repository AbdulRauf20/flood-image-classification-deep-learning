# ============================================================
# ResNet18 | FLOODNET | EXACT PAPER REPLICATION
# Paper : "Semi-Supervised Classification and Segmentation on
#          High Resolution Aerial Images" — Khose, Tiwari, Ghosh
#          (arXiv:2105.08655, NeurIPS 2021 CCML workshop)
# Target: Table 1 — ResNet18: 96.69% train acc, 96.70% test acc,
#          98.10% F1, 11.6M params
#
# This follows the authors' OFFICIAL code (FloodNet_T1.ipynb from
# github.com/sahilkhose/FloodNet), which differs from the paper text
# in several places. Where they differ, the CODE is followed:
#   * resnet18(pretrained=True), fc → Linear(512, 2), CrossEntropyLoss
#     (2 logits + argmax — NOT 1-logit BCE)
#   * SGD lr=0.01, NO momentum | batch 64
#   * images pre-resized to 400×300 with cv2 (W=400, H=300)
#   * augmentation: ONLY RandomHorizontalFlip + RandomVerticalFlip
#     (paper mentions crop/shift/resize but it is commented out in code)
#   * the 398 labeled images are split 50/50 stratified → 199 train,
#     199 valid. The authors had NO val/test labels locally.
#   * the unlabeled pool = Train/Unlabeled (1047) + official
#     Validation (450) + Test (448) images ≈ 1945 (transductive!)
#   * 200 epochs | alpha: 0 until ep 10, linear ramp → 1.0 at ep 150
#   * pseudo labels regenerated after every supervised phase (from
#     epoch 9), unlabeled loss = alpha * CE(logits, pseudo)
#   * best checkpoint selected by valid F1
#
# NOTE on "test accuracy": the paper's 96.70% was computed by the
# EarthVision challenge server from a submitted predictions JSON.
# Locally reproducible number = the 199-image valid split.
# This script also writes image_classes.json (same submission format)
# for the official Test images.
# ============================================================

import os, copy, json, random, time, warnings
warnings.filterwarnings("ignore")
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
import matplotlib.pyplot as plt

# ── 0. Knobs (exact from authors' code) ────────────────────────────────
MODEL_NAME      = "resnet18"  # "resnet18" = paper-exact | "vgg16" = lead's follow-up
SEED            = 42          # authors did NOT seed; we seed for reproducibility
RESIZE          = (400, 300)  # cv2 (width, height)
BATCH           = 64 if MODEL_NAME == "resnet18" else 16   # VGG16 @300×400 OOMs at 64
LR              = 0.01        # SGD, no momentum
EPOCHS          = 200
START_ALPHA     = 10          # alpha = 0 before this epoch
REACH_MAX_ALPHA = 150         # alpha reaches 1.0 here (linear ramp)
MAX_ALPHA       = 1.0

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ── 1. Locate the FloodNet Track-1 folder ──────────────────────────────
CANDIDATE_ROOTS = [
    "/kaggle/input/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge",
    "/kaggle/input/aerial-imagery-dataset-floodnet-challenge",
    "/kaggle/input",
]
BASE = None
for root in CANDIDATE_ROOTS:
    if not os.path.isdir(root):
        continue
    for dirpath, dirnames, _ in os.walk(root):
        if "Labeled" in dirnames and Path(dirpath).name == "Train":
            BASE = str(Path(dirpath).parent)
            break
    if BASE:
        break
if BASE is None:
    import kagglehub
    dl = kagglehub.dataset_download(
        "aletbm/aerial-imagery-dataset-floodnet-challenge")
    for dirpath, dirnames, _ in os.walk(dl):
        if "Labeled" in dirnames and Path(dirpath).name == "Train":
            BASE = str(Path(dirpath).parent)
            break
assert BASE is not None, "Could not locate the FloodNet Track-1 folder."
print(f"Dataset: {BASE}")

def get_images(folder):
    return sorted(
        glob.glob(os.path.join(folder, "*.jpg"))  +
        glob.glob(os.path.join(folder, "*.png"))  +
        glob.glob(os.path.join(folder, "*.jpeg"))
    )

FLOODED_DIR    = os.path.join(BASE, "Train", "Labeled", "Flooded",     "image")
NONFLOODED_DIR = os.path.join(BASE, "Train", "Labeled", "Non-Flooded", "image")
UNLABELED_DIR  = os.path.join(BASE, "Train", "Unlabeled", "image")
VAL_IMG_DIR    = os.path.join(BASE, "Validation", "image")
TEST_IMG_DIR   = os.path.join(BASE, "Test", "image")
# some copies have Validation/Validation/image etc. — fall back to a walk
if not os.path.isdir(VAL_IMG_DIR):
    for dirpath, dirnames, _ in os.walk(os.path.join(BASE, "Validation")):
        if "image" in dirnames: VAL_IMG_DIR = os.path.join(dirpath, "image"); break
if not os.path.isdir(TEST_IMG_DIR):
    for dirpath, dirnames, _ in os.walk(os.path.join(BASE, "Test")):
        if "image" in dirnames: TEST_IMG_DIR = os.path.join(dirpath, "image"); break

# ── 2. One-time resize cache (authors pre-resized to 400×300 on disk) ──
CACHE = os.path.join(OUT, "floodnet_resized_400x300")

def cached_path(src):
    rel = os.path.relpath(src, BASE)
    dst = os.path.join(CACHE, rel)
    return dst

def build_cache(paths, desc):
    t0, done = time.time(), 0
    for p in paths:
        dst = cached_path(p)
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img = cv2.imread(p)                       # BGR
        img = cv2.resize(img, RESIZE)             # (400, 300) = W×H
        cv2.imwrite(dst, img)
        done += 1
    print(f"  {desc:<28}: {len(paths):>5} images "
          f"({done} resized, {time.time()-t0:.0f}s)")

flooded_files    = get_images(FLOODED_DIR)
nonflooded_files = get_images(NONFLOODED_DIR)
unlabeled_files  = get_images(UNLABELED_DIR)
val_files        = get_images(VAL_IMG_DIR)
test_files       = get_images(TEST_IMG_DIR)

print("\nBuilding 400×300 resize cache (one-time) ...")
build_cache(flooded_files,    "Labeled Flooded")
build_cache(nonflooded_files, "Labeled Non-Flooded")
build_cache(unlabeled_files,  "Train Unlabeled")
build_cache(val_files,        "Official Validation imgs")
build_cache(test_files,       "Official Test imgs")

to_cached = lambda lst: [cached_path(p) for p in lst]
flooded_files, nonflooded_files = to_cached(flooded_files), to_cached(nonflooded_files)
unlabeled_files, val_files, test_files = \
    to_cached(unlabeled_files), to_cached(val_files), to_cached(test_files)

# ── 3. Labeled 50/50 stratified split (exact: train_test_split 0.5) ────
X = flooded_files + nonflooded_files
y = [1]*len(flooded_files) + [0]*len(nonflooded_files)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.5, stratify=y, random_state=SEED)

print(f"\nLabeled split: train {len(X_train)} "
      f"({sum(y_train)} flooded) | valid {len(X_valid)} "
      f"({sum(y_valid)} flooded)")

# unlabeled pool = train-unlabeled + official val + official test images
# (exactly like the authors — transductive pseudo-labeling)
unlabeled_pool = sorted(unlabeled_files + val_files + test_files)
pseudo_labels  = np.array([-1] * len(unlabeled_pool))
print(f"Unlabeled pool : {len(unlabeled_pool)} "
      f"({len(unlabeled_files)} train-unlab + {len(val_files)} val "
      f"+ {len(test_files)} test)")

# ── 4. Datasets / transforms (exact) ───────────────────────────────────
tfm = {
    "train": T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "valid": T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

class T1Dataset(Dataset):
    """Images from file list; returns (img, label) or (img, index)."""
    def __init__(self, X, y, transform):
        self.X, self.y, self.transform = X, y, transform
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.X[idx]), cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        if self.y is None:
            return img, idx          # index → pseudo_labels lookup
        return img, self.y[idx]

# weighted sampler over the 199 training images (class-balanced batches)
_, counts      = np.unique(y_train, return_counts=True)
class_weights  = [1.0/c for c in counts]
sample_weights = [class_weights[l] for l in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                replacement=True)

train_ldr = DataLoader(T1Dataset(X_train, y_train, tfm["train"]),
                       batch_size=BATCH, sampler=sampler,
                       num_workers=2, pin_memory=True)
valid_ldr = DataLoader(T1Dataset(X_valid, y_valid, tfm["valid"]),
                       batch_size=BATCH, shuffle=False,
                       num_workers=2, pin_memory=True)
unlab_ldr = DataLoader(T1Dataset(unlabeled_pool, None, tfm["valid"]),
                       batch_size=BATCH, shuffle=True,
                       num_workers=2, pin_memory=True)

# ── 5. Model: pretrained backbone, 2-logit head, CE, SGD ───────────────
#  resnet18 = exact paper model (11.6M params, Table 1)
#  vgg16    = same pipeline, swapped backbone (lead's follow-up question)
def build_model(name):
    if name == "resnet18":
        m = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "vgg16":
        m = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(4096, 2)
    else:
        raise ValueError(name)
    return m

model = build_model(MODEL_NAME).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n{MODEL_NAME}: {n_params/1e6:.2f}M params (paper ResNet18: 11.6M)")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# ── 6. Metrics helper ──────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    probs_all, preds_all, lbls_all = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out   = model(imgs.to(DEVICE))
            probs = torch.softmax(out, 1)[:, 1].cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            probs_all.extend(probs); preds_all.extend(preds)
            lbls_all.extend(np.asarray(lbls))
    acc  = accuracy_score(lbls_all, preds_all)
    f1   = f1_score(lbls_all,       preds_all, zero_division=0)
    prec = precision_score(lbls_all, preds_all, zero_division=0)
    rec  = recall_score(lbls_all,   preds_all, zero_division=0)
    try:    roc = roc_auc_score(lbls_all, probs_all)
    except: roc = float("nan")
    return acc, f1, prec, rec, roc

# ── 7. Semi-supervised training loop (exact ss_train logic) ────────────
alphas = np.linspace(0, MAX_ALPHA, REACH_MAX_ALPHA - START_ALPHA)

history = []
best_f1, best_epoch, best_state, best_metrics = -1, -1, None, None
CKPT = os.path.join(OUT, f"{MODEL_NAME}_floodnet_paper_best.pt")

print(f"\nTraining {EPOCHS} epochs | alpha 0 → {MAX_ALPHA} "
      f"(ep {START_ALPHA} → {REACH_MAX_ALPHA}) | SGD lr={LR} | batch {BATCH}")
print("="*88)
t_start = time.time()

for epoch in range(EPOCHS):
    t0 = time.time()
    if epoch < START_ALPHA:
        alpha = 0.0
    elif epoch - START_ALPHA >= len(alphas):
        alpha = alphas[-1]
    else:
        alpha = alphas[epoch - START_ALPHA]

    # ── phase 1: supervised on the 199 labeled train images ──
    model.train()
    sup_loss, n_b = 0.0, 0
    for imgs, lbls in train_ldr:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward(); optimizer.step()
        sup_loss += loss.item(); n_b += 1

    # ── regenerate pseudo labels (from epoch START_ALPHA-1, like authors)
    if epoch >= START_ALPHA - 1:
        model.eval()
        with torch.no_grad():
            for imgs, idxs in unlab_ldr:
                preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
                pseudo_labels[np.asarray(idxs)] = preds

    # ── phase 2: train on pseudo-labeled pool, loss scaled by alpha ──
    unl_loss, n_u = 0.0, 0
    if alpha > 0:
        model.train()
        for imgs, idxs in unlab_ldr:
            imgs = imgs.to(DEVICE)
            tgts = torch.tensor(pseudo_labels[np.asarray(idxs)],
                                dtype=torch.int64).to(DEVICE)
            optimizer.zero_grad()
            loss = alpha * criterion(model(imgs), tgts)
            loss.backward(); optimizer.step()
            unl_loss += loss.item(); n_u += 1

    # ── phase 3: validate on the held-out 199 labeled images ──
    acc, f1, prec, rec, roc = evaluate(model, valid_ldr)
    history.append(dict(epoch=epoch+1, alpha=round(alpha,4),
                        sup_loss=round(sup_loss/max(n_b,1),4),
                        unl_loss=round(unl_loss/max(n_u,1),4) if n_u else 0.0,
                        val_acc=round(acc,4), val_f1=round(f1,4),
                        val_precision=round(prec,4), val_recall=round(rec,4),
                        val_auc=round(roc,4)))

    flag = ""
    if f1 > best_f1:
        best_f1, best_epoch = f1, epoch + 1
        best_metrics = (acc, f1, prec, rec, roc)
        best_state = copy.deepcopy(model.state_dict())
        torch.save({"epoch": epoch, "model_state_dict": best_state}, CKPT)
        flag = "  ← best"
    print(f"Ep[{epoch+1:3d}/{EPOCHS}] α={alpha:.2f} │ "
          f"sup {sup_loss/max(n_b,1):.4f} │ "
          f"unl {unl_loss/max(n_u,1) if n_u else 0:.4f} │ "
          f"acc {acc*100:6.2f}% │ f1 {f1*100:6.2f}% │ "
          f"auc {roc:.4f} │ {time.time()-t0:.0f}s{flag}")

    # crash-safe history dump
    pd.DataFrame(history).to_csv(
        os.path.join(OUT, f"{MODEL_NAME}_floodnet_paper_history.csv"),
        index=False)

print("="*88)
print(f"Total time: {(time.time()-t_start)/3600:.2f} h")

# ── 8. Final report vs paper ───────────────────────────────────────────
model.load_state_dict(best_state)
acc, f1, prec, rec, roc = best_metrics

print("\n" + "═"*60)
print(f"  BEST {MODEL_NAME.upper()} (valid split, 199 images) "
      "vs PAPER Table 1 (ResNet18)")
print("═"*60)
print(f"  {'Metric':<12} {'Paper':>12} {'Ours (valid)':>14}")
print(f"  {'-'*44}")
print(f"  {'accuracy':<12} {'96.70%':>12} {acc*100:>13.2f}%")
print(f"  {'f1':<12} {'98.10%':>12} {f1*100:>13.2f}%")
print(f"  {'precision':<12} {'—':>12} {prec*100:>13.2f}%")
print(f"  {'recall':<12} {'—':>12} {rec*100:>13.2f}%")
print(f"  {'roc_auc':<12} {'—':>12} {roc*100:>13.2f}%")
print(f"  {'#params':<12} {'11.6M':>12} {n_params/1e6:>12.2f}M")
print(f"  best epoch : {best_epoch}")
print("═"*60)
print("  NOTE: paper 'test accuracy' was computed by the challenge")
print("  server (labels never public). Valid split is the local proxy.")

# ── 9. Submission-format predictions for official Test images ──────────
#  (same JSON format + label inversion the authors used)
test_ldr = DataLoader(T1Dataset(test_files, None, tfm["valid"]),
                      batch_size=BATCH, shuffle=False,
                      num_workers=2, pin_memory=True)
model.eval()
json_data = {}
with torch.no_grad():
    for imgs, idxs in test_ldr:
        preds = model(imgs.to(DEVICE)).argmax(1).cpu().tolist()
        for i, pr in zip(np.asarray(idxs), preds):
            stem = os.path.splitext(os.path.basename(test_files[i]))[0]
            json_data[stem] = pr ^ 1        # authors invert for submission
with open(os.path.join(OUT, "image_classes.json"), "w") as f:
    json.dump(json_data, f)
print(f"\nSaved → image_classes.json ({len(json_data)} test predictions)")

# ── 10. Training curves ────────────────────────────────────────────────
hist = pd.DataFrame(history)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"{MODEL_NAME} | FloodNet | paper-exact semi-supervised run",
             fontsize=13, fontweight="bold")

axes[0].plot(hist["epoch"], hist["sup_loss"], label="supervised loss")
axes[0].plot(hist["epoch"], hist["unl_loss"], label="unlabeled loss")
axes[0].set_xlabel("epoch"); axes[0].set_title("Loss"); axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(hist["epoch"], hist["val_acc"]*100, label="val acc %")
axes[1].plot(hist["epoch"], hist["val_f1"]*100,  label="val F1 %")
axes[1].axhline(96.70, ls="--", c="gray", label="paper acc 96.70")
axes[1].axhline(98.10, ls=":",  c="gray", label="paper F1 98.10")
axes[1].set_xlabel("epoch"); axes[1].set_title("Validation metrics")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

axes[2].plot(hist["epoch"], hist["alpha"], color="tab:red")
axes[2].set_xlabel("epoch"); axes[2].set_title("alpha schedule")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, f"{MODEL_NAME}_floodnet_paper_curves.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {MODEL_NAME}_floodnet_paper_curves.png")
print("\n✓ All done.")
