import os, copy, random, warnings
warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
import matplotlib.pyplot as plt

# ── 0. Experiment knobs ────────────────────────────────────────────────
SEED       = 42                    # single seed this time
LEVELS     = [0.20, 0.40, 0.60, 0.80, 0.90]   # 20 → 90
METHODS    = ["unstructured", "structured"]
E          = 50                    # dense training epochs
FT_EPOCHS  = 50                    # fine-tune budget = dense budget

# semi-supervised schedule — exact from the FloodNet paper / vgg-16.ipynb
E_ai   = 20     # supervised-only phase ends
E_af   = 40     # alpha ramp ends
A_I    = 0.0
A_F    = 1.0
LAMBDA = 0.2    # uncertainty offset (best in paper Table 1)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ── 1. PATHS — auto-detect the FloodNet layout ─────────────────────────
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
            BASE = str(Path(dirpath).parent)     # …/FloodNet Challenge - Track 1
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

LABELED_FLOODED    = os.path.join(BASE, "Train", "Labeled", "Flooded",     "image")
LABELED_NONFLOODED = os.path.join(BASE, "Train", "Labeled", "Non-Flooded", "image")
UNLABELED          = os.path.join(BASE, "Train", "Unlabeled", "image")
VAL_BASE           = os.path.join(BASE, "Validation")
TEST_BASE          = os.path.join(BASE, "Test")

print("=== Path Check ===")
for name, p in [("Labeled Flooded",    LABELED_FLOODED),
                ("Labeled NonFlooded", LABELED_NONFLOODED),
                ("Unlabeled",          UNLABELED),
                ("Validation",         VAL_BASE),
                ("Test",               TEST_BASE)]:
    print(f"  {'OK' if os.path.exists(p) else 'MISSING':7} {name:<20} → {p}")

# ── 2. Build image & label lists (identical to vgg-16.ipynb) ───────────
def get_images(folder):
    return sorted(
        glob.glob(os.path.join(folder, "*.jpg"))  +
        glob.glob(os.path.join(folder, "*.png"))  +
        glob.glob(os.path.join(folder, "*.jpeg"))
    )

def get_labeled_split(split_base):
    labeled = []
    flooded_dir    = os.path.join(split_base, "Flooded",     "image")
    nonflooded_dir = os.path.join(split_base, "Non-Flooded", "image")
    if os.path.exists(flooded_dir):
        labeled += [(p, 1) for p in get_images(flooded_dir)]
    if os.path.exists(nonflooded_dir):
        labeled += [(p, 0) for p in get_images(nonflooded_dir)]
    if not labeled:
        img_dir = os.path.join(split_base, "image")
        labeled = [(p, 1) for p in get_images(img_dir)]
    return labeled

flooded_imgs    = [(p, 1) for p in get_images(LABELED_FLOODED)]
nonflooded_imgs = [(p, 0) for p in get_images(LABELED_NONFLOODED)]
TRAIN_SAMPLES   = flooded_imgs + nonflooded_imgs
UNLABELED_PATHS = get_images(UNLABELED)
VAL_SAMPLES     = get_labeled_split(VAL_BASE)
TEST_SAMPLES    = get_labeled_split(TEST_BASE)

def class_counts(samples):
    f  = sum(1 for _, l in samples if l == 1)
    nf = sum(1 for _, l in samples if l == 0)
    return f, nf

print("\n=== Dataset Summary ===")
f, nf = class_counts(TRAIN_SAMPLES)
print(f"  Train labeled : {len(TRAIN_SAMPLES)} ({f} flooded | {nf} non-flooded)")
print(f"  Unlabeled     : {len(UNLABELED_PATHS)}")
print(f"  Validation    : {len(VAL_SAMPLES)}")
print(f"  Test          : {len(TEST_SAMPLES)}")

#  The two baseline runs (edit this list to split work across sessions):
#    supervised_only → plain supervised training on the 398 labeled chips
#    semi_supervised → paper Algorithm 1 (pseudo-labels, alpha ramp)
BASELINE_RUNS = ["supervised_only", "semi_supervised"]

# ── 3. Hyper-parameters ────────────────────────────────────────────────
IMG_SIZE = 224
BATCH    = 16
LR       = 1e-4
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_cfg = len(METHODS) * len(LEVELS)
print(f"\nDevice  : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"Plan    : {len(BASELINE_RUNS)} runs × seed {SEED} "
      f"× (1 dense@{E}ep + {n_cfg} pruned@{FT_EPOCHS}ep)")
print(f"Levels  : {[int(a*100) for a in LEVELS]}% | methods: {METHODS}")

# ── 4. Datasets / loaders ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

class LabeledDS(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        return transform(img), torch.tensor(lbl, dtype=torch.float32)

class UnlabeledDS(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return transform(img)

val_ldr  = DataLoader(LabeledDS(VAL_SAMPLES),  batch_size=BATCH,
                      shuffle=False, num_workers=2, pin_memory=True)
test_ldr = DataLoader(LabeledDS(TEST_SAMPLES), batch_size=BATCH,
                      shuffle=False, num_workers=2, pin_memory=True)
unlab_ldr = DataLoader(UnlabeledDS(UNLABELED_PATHS), batch_size=BATCH,
                       shuffle=False, num_workers=2, pin_memory=True)

def make_train_loader(samples):
    f, nf = class_counts(samples)
    n     = len(samples)
    cw    = [n/(2*nf), n/(2*f)]
    sw    = [cw[int(l)] for _, l in samples]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    return DataLoader(LabeledDS(samples), batch_size=BATCH, sampler=sampler,
                      num_workers=2, pin_memory=True)

# ── 5. Model builder (identical to the SEN1FLOODS11 pruning script) ────
def build_vgg16(pretrained=True):
    w = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.vgg16(weights=w)
    m.classifier[-1] = nn.Linear(4096, 1)   # 4096 → 1 (logits)
    return m

# ── 6. Params & FLOPs profiler (hook-based, no dependencies) ───────────
def profile_model(model, img_size=224):
    """Count parameters and MACs/FLOPs for one forward pass (batch=1)."""
    model = model.to("cpu").eval()
    macs_total = {"v": 0}
    per_layer  = {}
    hooks = []

    def conv_hook(name):
        def fn(mod, inp, out):
            k = mod.kernel_size[0] * mod.kernel_size[1]
            macs = out.numel() * (mod.in_channels // mod.groups) * k
            macs_total["v"] += macs
            per_layer[name] = macs
        return fn

    def lin_hook(name):
        def fn(mod, inp, out):
            macs = mod.in_features * mod.out_features
            macs_total["v"] += macs
            per_layer[name] = macs
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook(name)))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook(name)))

    with torch.no_grad():
        model(torch.randn(1, 3, img_size, img_size))
    for h in hooks: h.remove()

    n_params = sum(p.numel() for p in model.parameters())
    return n_params, macs_total["v"], per_layer

def count_nonzero_params(model):
    total, nonzero = 0, 0
    for p in model.parameters():
        total   += p.numel()
        nonzero += int((p != 0).sum().item())
    return total, nonzero

def effective_macs(model, per_layer_macs):
    """Theoretical MACs of surviving (nonzero) weights."""
    eff = 0
    named = dict(model.named_modules())
    for lname, macs in per_layer_macs.items():
        mod = named.get(lname)
        if mod is None:
            eff += macs; continue
        w = mod.weight.detach()
        frac = float((w != 0).sum().item()) / max(w.numel(), 1)
        eff += macs * frac
    return eff

# ── 7. Evaluation ──────────────────────────────────────────────────────
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

# ── 8. Semi-supervised helpers (paper Algorithm 1) ─────────────────────
def compute_alpha(epoch):
    if epoch < E_ai:
        return 0.0
    elif epoch < E_af:
        return (A_F - A_I) / (E_af - E_ai) * (epoch - E_ai) + A_I
    return 1.0

def assign_pseudo_labels(model, loader, lam):
    """Uncertainty-offset rule: keep only confident predictions (on CPU)."""
    model.eval()
    imgs_kept, lbls_kept = [], []
    with torch.no_grad():
        for imgs in loader:
            probs = torch.sigmoid(model(imgs.to(DEVICE)).squeeze(1)).cpu()
            for i, pv in enumerate(probs.tolist()):
                if pv <= (0.5 - lam):
                    imgs_kept.append(imgs[i]); lbls_kept.append(0)
                elif pv > (0.5 + lam):
                    imgs_kept.append(imgs[i]); lbls_kept.append(1)
    return imgs_kept, lbls_kept

# ── 9. Training / fine-tuning loop (keeps best-F1 checkpoint) ──────────
def train_loop(model, samples, epochs, semi=False, tag=""):
    """Plain supervised (semi=False) or paper Algorithm 1 (semi=True).
    Best checkpoint chosen by validation F1; returns val metrics + epoch."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    train_ldr = make_train_loader(samples)

    best_f1, best_epoch = -1, -1
    best_state, best_metrics = None, None
    for ep in range(epochs):
        alpha = compute_alpha(ep) if semi else 0.0
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, lbls in train_ldr:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), lbls)
            loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1

        pseudo_count = 0
        if semi and alpha > 0 and len(UNLABELED_PATHS) > 0:
            p_imgs, p_lbls = assign_pseudo_labels(model, unlab_ldr, LAMBDA)
            pseudo_count = len(p_imgs)
            if pseudo_count > 0:
                model.train()
                p_lbls = torch.tensor(p_lbls, dtype=torch.float32)
                for i in range(0, pseudo_count, BATCH):
                    xb = torch.stack(p_imgs[i:i+BATCH]).to(DEVICE)
                    yb = p_lbls[i:i+BATCH].to(DEVICE)
                    optimizer.zero_grad()
                    loss = alpha * criterion(model(xb).squeeze(1), yb)
                    loss.backward(); optimizer.step()
                    total_loss += loss.item(); n_batches += 1

        acc, f1, prec, rec, roc = evaluate(model, val_ldr)
        extra = f" │ α {alpha:.2f} │ pseudo {pseudo_count}" if semi else ""
        print(f"    {tag} ep {ep+1:2d}/{epochs} │ "
              f"loss {total_loss/max(n_batches,1):7.4f} │ acc {acc:.4f} │ "
              f"f1 {f1:.4f} │ auc {roc:.4f}{extra}")
        if f1 > best_f1:
            best_f1      = f1
            best_epoch   = ep + 1
            best_metrics = (acc, f1, prec, rec, roc)
            best_state   = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)      # keep the best checkpoint
    return best_metrics, best_epoch

# ── 10. Pruning utilities (identical to SEN1FLOODS11 script) ───────────
def prunable_modules(model):
    mods = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mods.append((name, mod))
    return mods

def apply_unstructured(model, amount):
    params = [(m, "weight") for _, m in prunable_modules(model)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                              amount=amount)

def apply_structured(model, amount):
    """L1 structured pruning: conv output filters + hidden-FC neurons
    (VGG-specific — the two 4096-wide FCs hold ~90% of the params).
    The final 1-node output layer is left intact."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.out_channels > 8:
            prune.ln_structured(mod, "weight", amount=amount, n=1, dim=0)
        elif isinstance(mod, nn.Linear) and mod.out_features > 8:
            prune.ln_structured(mod, "weight", amount=amount, n=1, dim=0)

def finalize_pruning(model):
    for _, mod in prunable_modules(model):
        if prune.is_pruned(mod):
            try: prune.remove(mod, "weight")
            except ValueError: pass

# ── 11. Dense checkpoint (train or reuse) ──────────────────────────────
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

def find_weights(fname):
    """Look for previously trained weights in Kaggle inputs / cwd."""
    for base in ["/kaggle/input", "."]:
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            if fname in filenames:
                return os.path.join(dirpath, fname)
    return None

def get_dense(run_name, semi):
    """Dense VGG-16 for a run: load checkpoint if attached as input
    (from a previous session), else train E epochs."""
    fname = f"vgg16_floodnet_dense_{run_name}_seed{SEED}.pth"
    path  = find_weights(fname)

    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    if path is not None:
        print(f"\n  Reusing dense checkpoint ← {path}")
        model = build_vgg16(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        metrics = evaluate(model, val_ldr)
        best_ep = -1     # unknown (from previous session)
    else:
        print(f"\n  Training dense | {run_name} | seed {SEED} "
              f"| {E} epochs ...")
        model = build_vgg16(pretrained=True).to(DEVICE)
        metrics, best_ep = train_loop(model, TRAIN_SAMPLES, E, semi=semi,
                                      tag=f"dense {run_name[:4]}")
        torch.save(model.state_dict(), f"{OUT}/{fname}")
        print(f"  Saved dense checkpoint → {fname}")
    return model, metrics, best_ep

# ── 12. THE PRUNING STUDY ──────────────────────────────────────────────
_ref = build_vgg16(pretrained=False)
DENSE_PARAMS, DENSE_MACS, PER_LAYER_MACS = profile_model(_ref)
del _ref
print(f"\nDense VGG-16: {DENSE_PARAMS/1e6:.2f}M params | "
      f"{2*DENSE_MACS/1e9:.2f} GFLOPs")

CONFIGS = [(m, a) for m in METHODS for a in LEVELS]
study_rows = []

def record(run, method, amount, best_ep, nonzero, eff, vmetrics, tmetrics):
    va, vf1, vp, vr, vroc = vmetrics
    ta, tf1, tp, tr, troc = tmetrics
    study_rows.append(dict(
        run=run, seed=SEED, method=method, amount=amount,
        best_epoch=best_ep, params_nonzero=nonzero,
        macs_effective=int(eff),
        val_accuracy=round(va,4),  val_f1=round(vf1,4),
        val_precision=round(vp,4), val_recall=round(vr,4),
        val_roc_auc=round(vroc,4),
        test_accuracy=round(ta,4),  test_f1=round(tf1,4),
        test_precision=round(tp,4), test_recall=round(tr,4),
        test_roc_auc=round(troc,4)))

for run_name in BASELINE_RUNS:
    semi = (run_name == "semi_supervised")
    print("\n\n" + "█"*88)
    print(f"  VGG-16 PRUNING  |  FLOODNET  |  {run_name}  |  seed {SEED}")
    print("█"*88)

    dense_model, dmetrics, dep = get_dense(run_name, semi)
    dtest = evaluate(dense_model, test_ldr)
    record(run_name, "dense", 0.0, dep, DENSE_PARAMS, DENSE_MACS,
           dmetrics, dtest)
    print(f"  DENSE │ val F1 {dmetrics[1]:.4f} │ test F1 {dtest[1]:.4f} "
          f"│ test AUC {dtest[4]:.4f}")

    dense_state = copy.deepcopy(dense_model.state_dict())
    del dense_model
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    for method, amount in CONFIGS:
        print(f"\n  ── {run_name} │ {method} @ {amount:.0%} │ "
              f"FT {FT_EPOCHS} ep ──")
        # re-seed so FT randomness is reproducible per config
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        model = build_vgg16(pretrained=False).to(DEVICE)
        model.load_state_dict(dense_state)

        if method == "unstructured":
            apply_unstructured(model, amount)
        else:
            apply_structured(model, amount)

        ft_metrics, ft_ep = train_loop(
            model, TRAIN_SAMPLES, FT_EPOCHS, semi=semi,
            tag=f"{method[:6]}@{amount:.0%}")
        finalize_pruning(model)
        ft_test = evaluate(model, test_ldr)

        total, nonzero = count_nonzero_params(model)
        eff = effective_macs(model, PER_LAYER_MACS)
        record(run_name, method, amount, ft_ep, nonzero, eff,
               ft_metrics, ft_test)

        print(f"    → NZ params {nonzero/1e6:.2f}M │ "
              f"eff FLOPs {2*eff/1e9:.2f}G │ val F1 {ft_metrics[1]:.4f} │ "
              f"test F1 {ft_test[1]:.4f} (best ep {ft_ep})")

        del model
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # incremental save after each run — crash-safe
    pd.DataFrame(study_rows).to_csv(
        f"{OUT}/vgg16_floodnet_pruning50_results.csv", index=False)
    print(f"\n  [checkpoint] results so far saved → "
          f"vgg16_floodnet_pruning50_results.csv")

# ── 13. Summary table ──────────────────────────────────────────────────
study_df = pd.DataFrame(study_rows)
study_df.to_csv(f"{OUT}/vgg16_floodnet_pruning50_results.csv", index=False)
print(f"\nSaved → vgg16_floodnet_pruning50_results.csv")

print("\n\n" + "═"*100)
print(f"  PRUNING SUMMARY — VGG-16 | FLOODNET | seed {SEED} | "
      f"FT {FT_EPOCHS} ep")
print("═"*100)
print(f"{'Run':<17} │ {'Method':<13} │ {'Amt':>4} │ {'NZ params':>10} │ "
      f"{'val F1':>7} │ {'test F1':>7} │ {'test Acc':>8} │ {'test AUC':>8}")
print("─"*100)
for _, r in study_df.iterrows():
    print(f"{r['run']:<17} │ {r['method']:<13} │ {r['amount']:>4.0%} │ "
          f"{r['params_nonzero']/1e6:>8.2f} M │ {r['val_f1']:>7.4f} │ "
          f"{r['test_f1']:>7.4f} │ {r['test_accuracy']:>8.4f} │ "
          f"{r['test_roc_auc']:>8.4f}")
print("═"*100)

# delta vs dense (single seed → no noise band, just raw deltas)
print("\n  ΔF1 (test) vs dense, same run:")
for run_name in BASELINE_RUNS:
    d = study_df[(study_df["run"] == run_name) &
                 (study_df["method"] == "dense")]
    if len(d) == 0: continue
    d_f1 = d.iloc[0]["test_f1"]
    for _, r in study_df[(study_df["run"] == run_name) &
                         (study_df["method"] != "dense")].iterrows():
        print(f"    {run_name:<17} {r['method']:<13} "
              f"@{r['amount']:>4.0%}  ΔF1 = {r['test_f1'] - d_f1:+.4f}")

# ── 14. Plot — F1 vs pruning level ─────────────────────────────────────
fig, axes = plt.subplots(1, len(BASELINE_RUNS), figsize=(15, 5.5),
                         squeeze=False)
fig.suptitle(f"VGG-16 | FloodNet | pruning study "
             f"(seed {SEED}, FT {FT_EPOCHS} ep)",
             fontsize=13, fontweight="bold")
mcolors = {"unstructured": "tab:blue", "structured": "tab:red"}

for ax, run_name in zip(axes.flat, BASELINE_RUNS):
    d = study_df[(study_df["run"] == run_name) &
                 (study_df["method"] == "dense")]
    if len(d) > 0:
        ax.axhline(d.iloc[0]["test_f1"], color="tab:green",
                   linewidth=1.6, label="dense")
    for method in METHODS:
        sub = study_df[(study_df["run"] == run_name) &
                       (study_df["method"] == method)].sort_values("amount")
        ax.plot([a*100 for a in sub["amount"]], sub["test_f1"],
                color=mcolors[method], marker="o", linewidth=1.6,
                label=method)
    ax.set_title(run_name, fontsize=11)
    ax.set_xlabel("Pruning amount (%)")
    ax.set_ylabel("Test F1 (best-val checkpoint)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/vgg16_floodnet_pruning50.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved → vgg16_floodnet_pruning50.png")

print("\n✓ All done.")
