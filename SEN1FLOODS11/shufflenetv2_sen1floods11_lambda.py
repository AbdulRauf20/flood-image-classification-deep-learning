#  SEN1FLOODS11 — ShuffleNet V2 (x1.0) Semi-Supervised Training

import os, copy, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ShuffleNet_V2_X1_0_Weights
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ── 1. PATHS — auto-detect the dataset layout ──────────────────────────
ROOT = Path("/kaggle/input/sen1floods11-essentials")
if not ROOT.exists():
    import kagglehub
    ROOT = Path(kagglehub.dataset_download("smabrarrajin/sen1floods11-essentials"))

def find_dir(root, target_name):
    """Walk directories only (cheap) and return the first match by name."""
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() == target_name.lower():
                return Path(dirpath) / d
    return None

S1_DIR    = find_dir(ROOT, "S1Hand")
LABEL_DIR = find_dir(ROOT, "LabelHand")
WEAK_DIR  = find_dir(ROOT, "S1Weak")          # weakly-labeled S1 chips

print("=== Path Check ===")
for name, p in [("S1Hand", S1_DIR), ("LabelHand", LABEL_DIR),
                ("Weak S1 (unlabeled pool)", WEAK_DIR)]:
    print(f"  {'OK' if p else 'MISSING':7} {name:<26} → {p}")
assert S1_DIR is not None and LABEL_DIR is not None, \
    "Could not locate S1Hand / LabelHand folders — check dataset root."

# ── 2. Derive image-level labels from pixel masks ──────────────────────
FLOOD_THRESHOLD    = 0.01   # chip = flooded if ≥1% of valid pixels are water
UNLABELED_FRACTION = 0.50   # fallback only (when no weak S1 folder exists)
MAX_UNLABELED      = 1500   # cap unlabeled pool (RAM guard, ≈ FloodNet size)

def water_fraction(mask_path):
    """Fraction of VALID pixels (mask != -1) that are water (mask == 1)."""
    with rasterio.open(mask_path) as src:
        m = src.read(1)
    valid = m != -1
    if not valid.any():
        return None
    return float((m[valid] == 1).mean())

print("\nDeriving image-level labels from masks "
      f"(flooded = water fraction ≥ {FLOOD_THRESHOLD:.2f}) ...")

all_samples = []                      # (s1_path, label)
for mask_path in sorted(LABEL_DIR.glob("*.tif")):
    stem    = mask_path.name.replace("_LabelHand.tif", "")
    s1_path = S1_DIR / f"{stem}_S1Hand.tif"
    if not s1_path.exists():
        continue
    wf = water_fraction(mask_path)
    if wf is None:                    # chip is entirely no-data
        continue
    all_samples.append((str(s1_path), int(wf >= FLOOD_THRESHOLD)))

flooded_s    = [s for s in all_samples if s[1] == 1]
nonflooded_s = [s for s in all_samples if s[1] == 0]
print(f"  Chips: {len(all_samples)}  "
      f"({len(flooded_s)} flooded | {len(nonflooded_s)} non-flooded)")

# ── 3. 80/20 labeled split (same as FloodNet run) ─────────────────────
random.shuffle(flooded_s)
random.shuffle(nonflooded_s)

def split80(lst):
    cut = int(0.8 * len(lst))
    return lst[:cut], lst[cut:]

f_train,  f_val  = split80(flooded_s)
nf_train, nf_val = split80(nonflooded_s)

TRAIN_SAMPLES = f_train + nf_train
VAL_SAMPLES   = f_val   + nf_val

# ── 4. Unlabeled pool ─────────────────────────────────────────────────
if WEAK_DIR is not None:
    UNLABELED_PATHS = [str(p) for p in sorted(WEAK_DIR.glob("*.tif"))]
    random.shuffle(UNLABELED_PATHS)
    UNLABELED_PATHS = UNLABELED_PATHS[:MAX_UNLABELED]
    unl_source = f"weakly-labeled S1 chips ({WEAK_DIR.name})"
else:
    # Standard SSL simulation: hide the labels of part of the train split
    random.shuffle(TRAIN_SAMPLES)
    cut             = int((1 - UNLABELED_FRACTION) * len(TRAIN_SAMPLES))
    UNLABELED_PATHS = [p for p, _ in TRAIN_SAMPLES[cut:]]
    TRAIN_SAMPLES   = TRAIN_SAMPLES[:cut]
    unl_source = (f"{UNLABELED_FRACTION:.0%} of train chips "
                  "(labels discarded — no weak S1 folder found)")

n_f_tr  = sum(1 for _, l in TRAIN_SAMPLES if l == 1)
n_nf_tr = sum(1 for _, l in TRAIN_SAMPLES if l == 0)
n_tr    = len(TRAIN_SAMPLES)

print(f"\n=== Dataset Split ===")
print(f"  Train     : {n_tr}  ({n_f_tr} flooded | {n_nf_tr} non-flooded)")
print(f"  Val       : {len(VAL_SAMPLES)}  "
      f"({sum(1 for _, l in VAL_SAMPLES if l == 1)} flooded | "
      f"{sum(1 for _, l in VAL_SAMPLES if l == 0)} non-flooded)")
print(f"  Unlabeled : {len(UNLABELED_PATHS)}  ← {unl_source}")

# ── 5. Fixed hyper-parameters (identical to FloodNet run) ─────────────
IMG_SIZE  = 224
BATCH     = 16
LR        = 1e-4
E         = 50
E_ia      = 20       # labeled-only phase ends   (E_i^a)
E_fa      = 40       # alpha ramp ends            (E_f^a)
a_i, a_f  = 0.0, 1.0
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# λ sweep: 0.0, 0.1, 0.2, ... 1.0  (11 values)
# FloodNet ShuffleNet V2 run used fixed λ=0.2 → use [0.2] to replicate it.
LAMBDA_VALUES = [round(x * 0.1, 1) for x in range(11)]

print(f"\nDevice  : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"λ sweep : {LAMBDA_VALUES}")
print(f"Total runs: {len(LAMBDA_VALUES)} × {E} epochs = {len(LAMBDA_VALUES)*E} epochs")

# ── 6. Alpha schedule (Algorithm 1, lines 2-7) ────────────────────────
def get_alpha(ep):
    if ep < E_ia: return a_i
    if ep < E_fa: return ((a_f-a_i)/(E_fa-E_ia))*(ep-E_ia)+a_i
    return a_f

# ── 7. SAR chip loading → 3-channel tensor ────────────────────────────
DB_MIN, DB_MAX = -50.0, 1.0     # typical Sentinel-1 backscatter range (dB)
IMAGENET_MEAN  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def load_s1_tensor(path):
    """2-band SAR GeoTIFF → normalized 3×224×224 float tensor."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)          # (2, H, W)
    arr = np.nan_to_num(arr, nan=DB_MIN)             # no-data → floor
    arr = np.clip(arr, DB_MIN, DB_MAX)
    arr = (arr - DB_MIN) / (DB_MAX - DB_MIN)         # dB → [0, 1]
    vv, vh = arr[0], arr[1]
    img = np.stack([vv, vh, (vv + vh) / 2.0])        # (3, H, W)
    t = torch.from_numpy(img)
    t = F.interpolate(t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                      mode="bilinear", align_corners=False).squeeze(0)
    return (t - IMAGENET_MEAN) / IMAGENET_STD

# ── 8. Dataset classes ────────────────────────────────────────────────
class LabeledDS(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        return load_s1_tensor(path), torch.tensor(lbl, dtype=torch.float32)

class UnlabeledDS(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return load_s1_tensor(self.paths[i]), self.paths[i]

# ── 9. Fixed dataloaders (same split for all λ runs) ──────────────────
cw      = [n_tr/(2*n_nf_tr), n_tr/(2*n_f_tr)]
sw      = [cw[int(s[1])] for s in TRAIN_SAMPLES]
sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

train_ds  = LabeledDS(TRAIN_SAMPLES)
val_ds    = LabeledDS(VAL_SAMPLES)
unl_ds    = UnlabeledDS(UNLABELED_PATHS)

train_ldr = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                       num_workers=2, pin_memory=True)
val_ldr   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                       num_workers=2, pin_memory=True)
unl_ldr   = DataLoader(unl_ds,   batch_size=BATCH, shuffle=False,
                       num_workers=2, pin_memory=True)

# ── 10. Build ShuffleNet V2 ───────────────────────────────────────────
#  ShuffleNet V2 uses channel splitting and shuffling for efficient
#  computation on mobile/embedded devices. x1.0 = standard width.
def build_shufflenet_v2():
    """
    ShuffleNet V2 (x1.0) with pretrained ImageNet weights.
    Replace final FC: 1024 → 1 (binary output, raw logits —
    BCEWithLogitsLoss applies the sigmoid internally, equivalent
    to the Sigmoid+BCELoss used in the FloodNet notebook).
    """
    m = models.shufflenet_v2_x1_0(
        weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
    )
    m.fc = nn.Linear(m.fc.in_features, 1)   # 1024 → 1
    return m

_probe = build_shufflenet_v2()
total_p = sum(p.numel() for p in _probe.parameters())
print(f"\n=== Building ShuffleNet V2 (x1.0) ===")
print(f"  Total params     : {total_p/1e6:.1f}M")
print(f"  Input            : 3 × {IMG_SIZE} × {IMG_SIZE}")
print(f"  Output           : 1 node (BCEWithLogitsLoss)")
del _probe

# ── 11. Evaluation ────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    probs_all, preds_all, lbls_all = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model(imgs.to(DEVICE)).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            probs_all.extend(probs)
            preds_all.extend(preds)
            lbls_all.extend(lbls.numpy().astype(int))
    acc  = accuracy_score(lbls_all, preds_all)
    f1   = f1_score(lbls_all,       preds_all, zero_division=0)
    prec = precision_score(lbls_all, preds_all, zero_division=0)
    rec  = recall_score(lbls_all,   preds_all, zero_division=0)
    try:    roc = roc_auc_score(lbls_all, probs_all)
    except: roc = float("nan")
    return acc, f1, prec, rec, roc

# ── 12. One full training run for a given λ ───────────────────────────
def train_one_lambda(lam):
    """Train ShuffleNet V2 for E epochs with uncertainty offset = lam."""
    model     = build_shufflenet_v2().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    history    = []
    best_f1    = -1
    best_epoch = -1
    best_row   = {}
    best_state = None

    print(f"\n{'═'*88}")
    print(f"  ShuffleNet V2  |  λ = {lam:.1f}  |  {E} epochs")
    print(f"{'═'*88}")
    print(f"{'Ep':>3} │ {'λ':>4} │ {'α':>5} │ {'Loss':>8} │ "
          f"{'Acc':>6} │ {'F1':>6} │ {'Prec':>6} │ {'Rec':>6} │ {'AUC':>6}")
    print(f"{'─'*88}")

    for ep in range(E):
        model.train()
        alpha = get_alpha(ep)
        total_loss, n_batches = 0.0, 0

        # Phase A — labeled pass
        for imgs, lbls in train_ldr:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), lbls)
            loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1

        # Phase B — pseudo-label unlabeled (Algorithm 1 lines 8-15)
        # λ=0.0 means boundary is exactly 0.5 — still runs but
        # only samples with p exactly ≤0.5 or ≥0.5 get labels.
        # Effectively labeled-only when λ=0 since p==0.5 is rare.
        if alpha > 0:
            model.eval()
            p_imgs, p_lbls = [], []
            with torch.no_grad():
                for imgs, _ in unl_ldr:
                    imgs  = imgs.to(DEVICE)
                    probs = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
                    for i, pv in enumerate(probs):
                        if pv <= 0.5 - lam:             # confident non-flooded
                            p_imgs.append(imgs[i].cpu())
                            p_lbls.append(torch.tensor(0.0))
                        elif pv >= 0.5 + lam:           # confident flooded
                            p_imgs.append(imgs[i].cpu())
                            p_lbls.append(torch.tensor(1.0))
                        # else: uncertain margin → ignored

            # Phase C — fine-tune on pseudo-labeled
            if p_imgs:
                model.train()
                for s in range(0, len(p_imgs), BATCH):
                    pi = torch.stack(p_imgs[s:s+BATCH]).to(DEVICE)
                    pl = torch.stack(p_lbls[s:s+BATCH]).to(DEVICE)
                    optimizer.zero_grad()
                    loss_u = alpha * criterion(model(pi).squeeze(1), pl)
                    loss_u.backward(); optimizer.step()
                    total_loss += loss_u.item(); n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        acc, f1, prec, rec, roc = evaluate(model, val_ldr)

        row = dict(lambda_=lam, epoch=ep+1, alpha=round(alpha,4),
                   loss=round(avg_loss,4), accuracy=round(acc,4),
                   f1=round(f1,4), precision=round(prec,4),
                   recall=round(rec,4), roc_auc=round(roc,4))
        history.append(row)

        print(f"{ep+1:3d} │ {lam:4.1f} │ {alpha:5.3f} │ {avg_loss:8.4f} │ "
              f"{acc:6.4f} │ {f1:6.4f} │ {prec:6.4f} │ {rec:6.4f} │ {roc:6.4f}")

        if f1 > best_f1:
            best_f1=f1; best_epoch=ep+1
            best_row=row.copy()
            best_state=copy.deepcopy(model.state_dict())

    print(f"{'─'*88}")
    print(f"  ★ Best epoch {best_epoch}  |  "
          f"Acc={best_row['accuracy']:.4f}  F1={best_row['f1']:.4f}  "
          f"Prec={best_row['precision']:.4f}  Rec={best_row['recall']:.4f}  "
          f"AUC={best_row['roc_auc']:.4f}")

    return history, best_row, best_state

# ── 13. λ SWEEP ───────────────────────────────────────────────────────
print("\n\n" + "█"*88)
print("  STARTING λ SWEEP  —  ShuffleNet V2  |  SEN1FLOODS11")
print("█"*88)

all_history   = []   # every epoch of every λ
summary_rows  = []   # one best row per λ
best_states   = {}   # λ → state_dict

for lam in LAMBDA_VALUES:
    hist, best_row, best_state = train_one_lambda(lam)
    all_history.extend(hist)
    summary_rows.append(best_row)
    best_states[lam] = best_state

# ── 14. Summary table ─────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("lambda_").reset_index(drop=True)

print("\n\n" + "═"*88)
print("  FINAL SUMMARY — Best results per λ value")
print("═"*88)
print(f"{'λ':>5} │ {'Epoch':>5} │ {'Acc':>6} │ {'F1':>6} │ "
      f"{'Prec':>6} │ {'Rec':>6} │ {'AUC':>6}")
print("─"*88)
for _, r in summary_df.iterrows():
    marker = "  ◄ BEST" if r["f1"] == summary_df["f1"].max() else ""
    print(f"{r['lambda_']:>5.1f} │ {r['epoch']:>5} │ {r['accuracy']:>6.4f} │ "
          f"{r['f1']:>6.4f} │ {r['precision']:>6.4f} │ "
          f"{r['recall']:>6.4f} │ {r['roc_auc']:>6.4f}{marker}")
print("═"*88)

best_lambda  = summary_df.loc[summary_df["f1"].idxmax(), "lambda_"]
best_overall = summary_df.loc[summary_df["f1"].idxmax()]
print(f"\n  Best λ overall : {best_lambda}")
print(f"  Best Accuracy  : {best_overall['accuracy']:.4f}")
print(f"  Best F1        : {best_overall['f1']:.4f}")
print(f"  Best Precision : {best_overall['precision']:.4f}")
print(f"  Best Recall    : {best_overall['recall']:.4f}")
print(f"  Best ROC-AUC   : {best_overall['roc_auc']:.4f}")
print(f"  Best Epoch     : {int(best_overall['epoch'])}")

# ── 15. Save ──────────────────────────────────────────────────────────
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
all_df = pd.DataFrame(all_history)
all_df.to_csv(f"{OUT}/shufflenetv2_sen1floods11_all_history.csv", index=False)
summary_df.to_csv(f"{OUT}/shufflenetv2_sen1floods11_lambda_summary.csv", index=False)
torch.save(best_states[best_lambda],
           f"{OUT}/shufflenetv2_sen1floods11_best.pth")
print(f"\nSaved → shufflenetv2_sen1floods11_all_history.csv")
print(f"Saved → shufflenetv2_sen1floods11_lambda_summary.csv")
print(f"Saved → shufflenetv2_sen1floods11_best.pth  (λ={best_lambda})")

# ── 16. Plot 1 — metrics vs λ (summary) ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("ShuffleNet V2 | SEN1FLOODS11 | Best metrics vs λ",
             fontsize=13, fontweight="bold")
pairs = [("accuracy","Accuracy","tab:blue"),
         ("f1","F1 Score","tab:green"),
         ("precision","Precision","tab:orange"),
         ("recall","Recall","tab:purple"),
         ("roc_auc","ROC-AUC","tab:brown"),
         ("epoch","Best Epoch","tab:red")]

for ax, (col, title, color) in zip(axes.flat, pairs):
    ax.plot(summary_df["lambda_"], summary_df[col],
            color=color, linewidth=2, marker="o", markersize=6)
    best_val = summary_df[col].max() if col != "epoch" else None
    if best_val is not None:
        best_lam = summary_df.loc[summary_df[col].idxmax(), "lambda_"]
        ax.axvline(best_lam, color="black", linestyle="--",
                   linewidth=1, alpha=0.6, label=f"Best λ={best_lam}")
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("λ (uncertainty offset)")
    ax.set_ylabel(title)
    ax.set_xticks(LAMBDA_VALUES)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/shufflenetv2_sen1floods11_metrics_vs_lambda.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved → shufflenetv2_sen1floods11_metrics_vs_lambda.png")

# ── 17. Plot 2 — training curves for best λ ───────────────────────────
best_hist = all_df[all_df["lambda_"] == best_lambda].reset_index(drop=True)
best_ep   = int(best_overall["epoch"])

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
fig2.suptitle(
    f"ShuffleNet V2 | SEN1FLOODS11 | λ={best_lambda} (best) | Best epoch={best_ep}",
    fontsize=13, fontweight="bold"
)
for ax, (col, title, color) in zip(axes2.flat, pairs[:-1]):
    ax.plot(best_hist["epoch"], best_hist[col], color=color, linewidth=2)
    ax.axvline(best_ep, color="black", linestyle="--",
               linewidth=1.2, label=f"Best ep {best_ep}")
    bv = best_hist.loc[best_hist["epoch"]==best_ep, col].values[0]
    ax.scatter([best_ep],[bv], color="black", zorder=5, s=60)
    ax.axvline(E_ia, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(E_fa, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Last panel: all λ F1 curves together
ax_last = axes2.flat[-1]
for lam in LAMBDA_VALUES:
    sub = all_df[all_df["lambda_"]==lam]
    ax_last.plot(sub["epoch"], sub["f1"],
                 linewidth=1.2, alpha=0.7, label=f"λ={lam}")
ax_last.set_title("F1 — all λ values", fontsize=11)
ax_last.set_xlabel("Epoch")
ax_last.set_ylabel("F1 Score")
ax_last.legend(fontsize=6, ncol=2)
ax_last.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/shufflenetv2_sen1floods11_training_curves.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved → shufflenetv2_sen1floods11_training_curves.png")

print("\n✓ All done.")
