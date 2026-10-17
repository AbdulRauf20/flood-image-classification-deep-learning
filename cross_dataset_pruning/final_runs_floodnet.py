# ============================================================
# FINAL RUNS | FLOODNET | ResNet18 + VGG-16 | 3 seeds
# Lead's final protocol:
#   * FINAL-EPOCH checkpoint reported (NOT best-val-F1)
#   * metrics: accuracy, balanced accuracy, macro-F1, AUC
#     (binary F1/precision/recall also logged for continuity)
#   * seeds 42 / 124 / 2024 → mean ± std
#   * every pruned run starts from the dense model of the SAME seed
#   * experiment groups in this file (per dataset):
#       1. dense            : 2 archs × 3 seeds
#       2. structured       : 30/50/70/90% × 2 archs × 3 seeds (FT lr 0.01)
#       3. structured lr-ablation (FT lr 0.001):
#            ResNet18 @ 50/90%, VGG-16 @ 70%, seeds 42/124
#       4. unstructured     : 50/90% × 2 archs × 3 seeds (FT lr 0.01)
#   * split is FIXED (data seed 42) — only training randomness varies
#   * RESUMABLE: dense ckpts reused per (arch, seed); result rows already
#     present in an attached final_runs CSV are skipped
# ============================================================

import os, copy, glob, random, time, warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

# ── 0. Protocol knobs ──────────────────────────────────────────────────
DATASET          = "floodnet"
DATA_SEED        = 42                      # fixed split — never changes
RUN_SEEDS        = [42, 124, 2024]
ARCHS            = ["resnet18", "vgg16"]   # trim to split across sessions
STRUCT_LEVELS    = [0.30, 0.50, 0.70, 0.90]
UNSTRUCT_LEVELS  = [0.50, 0.90]
FT_LR_MAIN       = 0.01
FT_LR_ABLATION   = 0.001
ABLATION_CONFIGS = {"resnet18": [0.50, 0.90], "vgg16": [0.70]}
ABLATION_SEEDS   = [42, 124]
E                = 50                      # dense epochs
FT_EPOCHS        = 50                      # fine-tune epochs
IMG_W, IMG_H     = 400, 300

def batch_for(arch): return 64 if arch == "resnet18" else 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
PERSEED_CSV = f"final_runs_{DATASET}_perseed.csv"
AGG_CSV     = f"final_runs_{DATASET}_aggregate.csv"
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda": print(f"GPU   : {torch.cuda.get_device_name(0)}")

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ── 1. Locate FloodNet Track-1 ─────────────────────────────────────────
CANDIDATE_ROOTS = [
    "/kaggle/input/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge",
    "/kaggle/input/aerial-imagery-dataset-floodnet-challenge",
    "/kaggle/input",
]
BASE = None
for root in CANDIDATE_ROOTS:
    if not os.path.isdir(root): continue
    for dirpath, dirnames, _ in os.walk(root):
        if "Labeled" in dirnames and Path(dirpath).name == "Train":
            BASE = str(Path(dirpath).parent); break
    if BASE: break
if BASE is None:
    import kagglehub
    dl = kagglehub.dataset_download(
        "aletbm/aerial-imagery-dataset-floodnet-challenge")
    for dirpath, dirnames, _ in os.walk(dl):
        if "Labeled" in dirnames and Path(dirpath).name == "Train":
            BASE = str(Path(dirpath).parent); break
assert BASE is not None, "Could not locate the FloodNet Track-1 folder."

def get_images(folder):
    return sorted(glob.glob(os.path.join(folder, "*.jpg"))  +
                  glob.glob(os.path.join(folder, "*.png"))  +
                  glob.glob(os.path.join(folder, "*.jpeg")))

FLOODED_DIR    = os.path.join(BASE, "Train", "Labeled", "Flooded",     "image")
NONFLOODED_DIR = os.path.join(BASE, "Train", "Labeled", "Non-Flooded", "image")

# ── 2. One-time 400×300 resize cache ───────────────────────────────────
CACHE = os.path.join(OUT, "floodnet_resized_400x300")
def cached_path(src): return os.path.join(CACHE, os.path.relpath(src, BASE))
def build_cache(paths):
    for p in paths:
        dst = cached_path(p)
        if os.path.exists(dst): continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, cv2.resize(cv2.imread(p), (IMG_W, IMG_H)))

flooded_files    = get_images(FLOODED_DIR)
nonflooded_files = get_images(NONFLOODED_DIR)
print("Building 400×300 resize cache (one-time) ...")
build_cache(flooded_files); build_cache(nonflooded_files)
flooded_files    = [cached_path(p) for p in flooded_files]
nonflooded_files = [cached_path(p) for p in nonflooded_files]

# ── 3. FIXED 50/50 stratified split (data seed 42, all runs) ───────────
X = flooded_files + nonflooded_files
y = [1]*len(flooded_files) + [0]*len(nonflooded_files)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.5, stratify=y, random_state=DATA_SEED)
print(f"Split: train {len(X_train)} ({sum(y_train)} flooded) | "
      f"valid {len(X_valid)} ({sum(y_valid)} flooded)")
assert 0 < sum(y_valid) < len(y_valid), "Val must contain BOTH classes."

# ── 4. Datasets / loaders ──────────────────────────────────────────────
tfm_train = T.Compose([
    T.ToPILImage(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
tfm_valid = T.Compose([
    T.ToPILImage(), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ImgDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X, self.y, self.transform = X, y, transform
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.X[idx]), cv2.COLOR_BGR2RGB)
        return self.transform(img), self.y[idx]

def make_loaders(batch):
    _, counts      = np.unique(y_train, return_counts=True)
    class_weights  = [1.0/c for c in counts]
    sample_weights = [class_weights[l] for l in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)
    tr = DataLoader(ImgDataset(X_train, y_train, tfm_train),
                    batch_size=batch, sampler=sampler,
                    num_workers=2, pin_memory=True)
    va = DataLoader(ImgDataset(X_valid, y_valid, tfm_valid),
                    batch_size=batch, shuffle=False,
                    num_workers=2, pin_memory=True)
    return tr, va

# ── 5. Models / profiler / pruning (same operators as before) ─────────
def build_model(arch, pretrained=True):
    if arch == "resnet18":
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        w = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.vgg16(weights=w)
        m.classifier[-1] = nn.Linear(4096, 2)
    return m

def count_nonzero_params(model):
    total, nonzero = 0, 0
    for p in model.parameters():
        total += p.numel(); nonzero += int((p != 0).sum().item())
    return total, nonzero

def prunable_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]

def apply_unstructured(model, amount):
    params = [(m, "weight") for _, m in prunable_modules(model)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                              amount=amount)

def apply_structured(model, amount):
    for _, mod in prunable_modules(model):
        if isinstance(mod, nn.Conv2d) and mod.out_channels > 8:
            prune.ln_structured(mod, "weight", amount=amount, n=1, dim=0)
        elif isinstance(mod, nn.Linear) and mod.out_features > 8:
            prune.ln_structured(mod, "weight", amount=amount, n=1, dim=0)

def finalize_pruning(model):
    for _, mod in prunable_modules(model):
        if prune.is_pruned(mod):
            try: prune.remove(mod, "weight")
            except ValueError: pass

# ── 6. Evaluation (lead's metric set) ──────────────────────────────────
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
    acc     = accuracy_score(lbls_all, preds_all)
    bal_acc = balanced_accuracy_score(lbls_all, preds_all)
    mac_f1  = f1_score(lbls_all, preds_all, average="macro", zero_division=0)
    bin_f1  = f1_score(lbls_all, preds_all, zero_division=0)
    prec    = precision_score(lbls_all, preds_all, zero_division=0)
    rec     = recall_score(lbls_all,   preds_all, zero_division=0)
    try:    auc = roc_auc_score(lbls_all, probs_all)
    except: auc = float("nan")
    return dict(accuracy=acc, balanced_accuracy=bal_acc, macro_f1=mac_f1,
                binary_f1=bin_f1, precision=prec, recall=rec, auc=auc)

# ── 7. Training loop — FINAL-EPOCH model (no best-ckpt selection) ──────
def train_loop(model, epochs, lr, batch, tag=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_ldr, valid_ldr = make_loaders(batch)
    for ep in range(epochs):
        model.train()
        total_loss, n_b = 0.0, 0
        for imgs, lbls in train_ldr:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward(); optimizer.step()
            total_loss += loss.item(); n_b += 1
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            m = evaluate(model, valid_ldr)
            print(f"    {tag} ep {ep+1:2d}/{epochs} │ "
                  f"loss {total_loss/max(n_b,1):7.4f} │ "
                  f"acc {m['accuracy']:.4f} │ balacc "
                  f"{m['balanced_accuracy']:.4f} │ "
                  f"mF1 {m['macro_f1']:.4f} │ auc {m['auc']:.4f}")
    _, valid_ldr = make_loaders(batch)
    return evaluate(model, valid_ldr)        # FINAL-epoch metrics

# ── 8. Resume support ──────────────────────────────────────────────────
def find_file(fname):
    for base in ["/kaggle/input", OUT, "."]:
        if not os.path.isdir(base): continue
        for dirpath, _, filenames in os.walk(base):
            if fname in filenames:
                return os.path.join(dirpath, fname)
    return None

prev = find_file(PERSEED_CSV)
study_rows = []
done = set()
if prev is not None:
    old = pd.read_csv(prev)
    study_rows = old.to_dict("records")
    done = {(r["arch"], int(r["seed"]), r["method"],
             round(float(r["amount"]), 2), float(r["ft_lr"]))
            for r in study_rows}
    print(f"Resuming: {len(done)} configs already done ← {prev}")

def key(arch, seed, method, amount, ft_lr):
    return (arch, int(seed), method, round(float(amount), 2), float(ft_lr))

def record(arch, seed, method, amount, ft_lr, nonzero, metrics):
    study_rows.append(dict(
        arch=arch, dataset=DATASET, seed=seed, method=method,
        amount=amount, ft_lr=ft_lr, params_nonzero=nonzero,
        **{k: round(v, 4) for k, v in metrics.items()}))
    pd.DataFrame(study_rows).to_csv(os.path.join(OUT, PERSEED_CSV),
                                    index=False)

# ── 9. THE RUNS ────────────────────────────────────────────────────────
for seed in RUN_SEEDS:
    for arch in ARCHS:
        batch = batch_for(arch)
        print("\n\n" + "█"*80)
        print(f"  {arch} | {DATASET} | seed {seed}")
        print("█"*80)

        # ---- dense (final-epoch checkpoint, per seed) ----
        dense_fname = f"{arch}_{DATASET}_dense_final_seed{seed}.pth"
        dense_path  = find_file(dense_fname)
        seed_everything(seed)
        if dense_path is not None:
            print(f"  Reusing dense ← {dense_path}")
            dense_model = build_model(arch, pretrained=False).to(DEVICE)
            dense_model.load_state_dict(
                torch.load(dense_path, map_location=DEVICE))
            if key(arch, seed, "dense", 0.0, FT_LR_MAIN) not in done:
                _, va = make_loaders(batch)
                dmetrics = evaluate(dense_model, va)
                _, nz = count_nonzero_params(dense_model)
                record(arch, seed, "dense", 0.0, FT_LR_MAIN, nz, dmetrics)
        else:
            print(f"  Training dense | {E} ep | lr {FT_LR_MAIN} ...")
            dense_model = build_model(arch, pretrained=True).to(DEVICE)
            dmetrics = train_loop(dense_model, E, FT_LR_MAIN, batch,
                                  tag=f"dense s{seed}")
            torch.save(dense_model.state_dict(),
                       os.path.join(OUT, dense_fname))
            print(f"  Saved dense → {dense_fname}")
            if key(arch, seed, "dense", 0.0, FT_LR_MAIN) not in done:
                _, nz = count_nonzero_params(dense_model)
                record(arch, seed, "dense", 0.0, FT_LR_MAIN, nz, dmetrics)

        dense_state = copy.deepcopy(dense_model.state_dict())
        del dense_model
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

        # ---- pruned configs for this (arch, seed) ----
        configs  = [("structured", a, FT_LR_MAIN) for a in STRUCT_LEVELS]
        configs += [("unstructured", a, FT_LR_MAIN) for a in UNSTRUCT_LEVELS]
        if seed in ABLATION_SEEDS:
            configs += [("structured", a, FT_LR_ABLATION)
                        for a in ABLATION_CONFIGS[arch]]

        for method, amount, ft_lr in configs:
            if key(arch, seed, method, amount, ft_lr) in done:
                print(f"  [skip, done] {method}@{amount:.0%} lr={ft_lr}")
                continue
            print(f"\n  ── {arch} │ s{seed} │ {method} @ {amount:.0%} │ "
                  f"FT lr {ft_lr} ──")
            seed_everything(seed)
            model = build_model(arch, pretrained=False).to(DEVICE)
            model.load_state_dict(dense_state)   # same-seed dense start
            if method == "unstructured":
                apply_unstructured(model, amount)
            else:
                apply_structured(model, amount)

            metrics = train_loop(model, FT_EPOCHS, ft_lr, batch,
                                 tag=f"{method[:6]}@{amount:.0%} s{seed}")
            finalize_pruning(model)
            _, nonzero = count_nonzero_params(model)
            record(arch, seed, method, amount, ft_lr, nonzero, metrics)
            print(f"    → NZ {nonzero/1e6:.2f}M │ "
                  f"acc {metrics['accuracy']:.4f} │ "
                  f"balacc {metrics['balanced_accuracy']:.4f} │ "
                  f"mF1 {metrics['macro_f1']:.4f} │ "
                  f"auc {metrics['auc']:.4f}")
            del model
            if DEVICE.type == "cuda": torch.cuda.empty_cache()

# ── 10. Aggregate mean ± std across seeds ──────────────────────────────
df = pd.DataFrame(study_rows)
df.to_csv(os.path.join(OUT, PERSEED_CSV), index=False)

agg_rows = []
for (arch, method, amount, ft_lr), sub in df.groupby(
        ["arch", "method", "amount", "ft_lr"]):
    row = dict(arch=arch, dataset=DATASET, method=method, amount=amount,
               ft_lr=ft_lr, n_seeds=len(sub),
               params_nonzero=int(sub["params_nonzero"].mean()))
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "auc"]:
        row[f"{metric}_mean"] = round(sub[metric].mean(), 4)
        row[f"{metric}_std"]  = round(sub[metric].std(ddof=0)
                                      if len(sub) > 1 else 0.0, 4)
    agg_rows.append(row)
agg = pd.DataFrame(agg_rows).sort_values(
    ["arch", "method", "ft_lr", "amount"])
agg.to_csv(os.path.join(OUT, AGG_CSV), index=False)

print("\n\n" + "═"*110)
print(f"  FINAL-EPOCH RESULTS — {DATASET.upper()} | mean ± std over seeds")
print("═"*110)
print(f"{'Arch':<9} │ {'Method':<13} │ {'Amt':>4} │ {'lr':>6} │ {'n':>2} │ "
      f"{'Acc':>15} │ {'BalAcc':>15} │ {'macroF1':>15} │ {'AUC':>15}")
print("─"*110)
for _, r in agg.iterrows():
    print(f"{r['arch']:<9} │ {r['method']:<13} │ {r['amount']:>4.0%} │ "
          f"{r['ft_lr']:>6} │ {r['n_seeds']:>2} │ "
          f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} │ "
          f"{r['balanced_accuracy_mean']:.4f}±{r['balanced_accuracy_std']:.4f} │ "
          f"{r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} │ "
          f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}")
print("═"*110)
print(f"\nSaved → {PERSEED_CSV}")
print(f"Saved → {AGG_CSV}")
print("\n✓ All done.")
