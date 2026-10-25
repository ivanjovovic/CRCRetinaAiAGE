import os, sys, json, math, time, copy, random, argparse, hashlib, platform, contextlib
import importlib.metadata
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms

import timm

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt




def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_everything(seed: int, deterministic: bool = True):
    set_seed(int(seed))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p


def new_output(path) -> Path:
    return ensure_dir(Path(path).expanduser().resolve())


def save_json(obj, path: Path):
    def conv(x):
        if isinstance(x, dict):  return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [conv(v) for v in x]
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.integer,)):  return int(x)
        return x
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv(obj), f, indent=2, ensure_ascii=False)


def load_config(path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def environment() -> Dict:
    pk = {}
    for n in ["torch","torchvision","timm","numpy","pandas","scipy","scikit-learn","Pillow","matplotlib"]:
        try: pk[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: pk[n] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": pk,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def autocast(device, enabled=True):
    if device.type != "cuda": return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", enabled=bool(enabled))


IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS])




class ListDataset(Dataset):

    def __init__(self, items: List[Tuple[Path, int]], tf, mirror: bool = False):
        self.tf = tf
        self.items = [(p, y, f) for p, y in items for f in ([False, True] if mirror else [False])]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y, flip = self.items[i]
        img = Image.open(p).convert("RGB")
        if flip: img = ImageOps.mirror(img)
        return self.tf(img), y


def participant_level_split(root: Path, class_names: List[str], pos_class: str,
                            ratios: List[float], seed: int) -> Dict[str, List[Tuple[Path,int]]]:

    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rnd = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    stats = {}
    for c in class_names:
        y = 1 if c == pos_class else 0
        imgs = list_images(root / c)
        groups: Dict[str, List[Path]] = {}
        for p in imgs:
            pid = p.stem.split("_")[0]
            groups.setdefault(pid, []).append(p)
        gids = list(groups.keys()); rnd.shuffle(gids)
        n = len(gids)
        n_tr = int(round(ratios[0] * n)); n_va = int(round(ratios[1] * n))
        assign = {"train": gids[:n_tr], "val": gids[n_tr:n_tr+n_va], "test": gids[n_tr+n_va:]}
        for sp, ids in assign.items():
            for gid in ids:
                for p in groups[gid]:
                    splits[sp].append((p, y))
        stats[c] = {sp: sum(1 for _ in assign[sp]) for sp in assign}
    for sp in splits: rnd.shuffle(splits[sp])
    return splits, stats


def write_manifest(splits, path: Path):
    rows = [{"participant_id": p.stem.split("_")[0], "image_path": str(p), "label": y, "split": sp,
             "image_sha256": sha256(p)} for sp in ["train","val","test"] for p, y in splits[sp]]
    df = pd.DataFrame(rows)
    if df.participant_id.duplicated().any():
        raise ValueError("Expected one image per participant.")
    df.to_csv(path, index=False)
    return df


def read_manifest(path, check_files=False):
    df = pd.read_csv(path, dtype={"participant_id": str})
    if not {"participant_id","image_path","label","split"} <= set(df):
        raise ValueError("Manifest needs participant_id, image_path, label, split.")
    if df.participant_id.duplicated().any():
        raise ValueError("Duplicate participant IDs.")
    if check_files:
        for p in df.image_path:
            if not Path(p).is_file(): raise FileNotFoundError(p)
    return df


def assert_split_ready(df):
    if set(df.split) != {"train","val","test"}:
        raise ValueError("Manifest split must contain train/val/test.")


def verify_images(df):
    if "image_sha256" in df:
        for row in df.itertuples():
            if sha256(row.image_path) != row.image_sha256:
                raise ValueError(f"Image changed: {row.participant_id}")


def load_covariates(path, manifest):
    df = pd.read_csv(path, dtype={"participant_id": str}, keep_default_na=False, na_values=[""])
    if not {"participant_id","age","vascular"} <= set(df):
        raise ValueError("Missing covariate columns.")
    if df.participant_id.duplicated().any():
        raise ValueError("Duplicate covariate IDs.")
    if "label" in df or "split" in df:
        raise ValueError("label/split come from the imaging manifest.")
    if set(df.participant_id) - set(manifest.participant_id):
        raise ValueError("Covariate IDs absent from manifest.")
    return manifest[["participant_id","label","split"]].merge(df, on="participant_id", how="left", validate="one_to_one")


def build_transforms(C):
    H, W = int(C["data"]["height"]), int(C["data"]["width"])
    ops_common = [transforms.Resize((H, W))]
    if C["data"].get("grayscale_3ch", True):
        ops_common.append(transforms.Grayscale(num_output_channels=3))

    train_ops = list(ops_common)
    if C["data"].get("hflip_train_only", True) and not C["training"].get("mirror_views", False):
        train_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    train_ops.append(transforms.ToTensor())

    eval_ops = list(ops_common) + [transforms.ToTensor()]

    if C["data"].get("imagenet_normalize", False):
        norm_ = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        train_ops.append(norm_); eval_ops.append(norm_)

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_model(C, pretrained=True):
    return timm.create_model(C["model"]["backbone"], pretrained=pretrained,
                             num_classes=len(C["data"]["class_names"]), drop_rate=float(C["model"]["dropout"]))


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model = build_model(state["config"], pretrained=False)
    model.load_state_dict(state["model"]); model.to(device).eval()
    return model, state




@torch.no_grad()
def collect_scores(model, loader, device, train_mode=False):
    model.train() if train_mode else model.eval()
    logits_all, y_all, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", enabled=(device.type=="cuda")):
            logits = model(x)
            loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0); n += x.size(0)
        logits_all.append(logits.float().cpu().numpy()); y_all.append(y.cpu().numpy())
    logits_all = np.concatenate(logits_all); y_all = np.concatenate(y_all)
    probs = torch.softmax(torch.from_numpy(logits_all), dim=1).numpy()
    return y_all, probs, loss_sum / max(1, n)


def youden_threshold(y_true_bin, y_score_pos) -> float:
    fpr, tpr, thr = roc_curve(y_true_bin, y_score_pos, pos_label=1)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))]) if len(thr) else 0.5


def hanley_mcneil(auc, n_pos, n_neg) -> Dict:
    q1, q2 = auc/(2-auc), 2*auc*auc/(1+auc)
    se = math.sqrt(max(0.0, (auc*(1-auc) + (n_pos-1)*(q1-auc*auc) + (n_neg-1)*(q2-auc*auc)) / (n_pos*n_neg)))
    return {"se": se, "low": max(0.0, auc-1.96*se), "high": min(1.0, auc+1.96*se)}


def power_auroc(auc, n_pos, n_neg, alpha=0.05) -> float:
    se0 = math.sqrt((n_pos+n_neg+1) / (12*n_pos*n_neg))
    se1 = hanley_mcneil(auc, n_pos, n_neg)["se"]
    z = norm.ppf(1 - alpha/2)
    return float(norm.sf((0.5 + z*se0 - auc)/se1) + norm.cdf((0.5 - z*se0 - auc)/se1))


def metrics_at_threshold(y_true_bin, y_score_pos, thr) -> Dict:
    auroc = roc_auc_score(y_true_bin, y_score_pos)
    ap = average_precision_score(y_true_bin, y_score_pos)
    pred = (y_score_pos >= thr).astype(int)
    acc = float((pred == y_true_bin).mean())
    f1m = float(f1_score(y_true_bin, pred, average="macro"))
    tn, fp, fn, tp = confusion_matrix(y_true_bin, pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp+fn) else float("nan")
    spec = tn / (tn + fp) if (tn+fp) else float("nan")
    ppv  = tp / (tp + fp) if (tp+fp) else float("nan")
    npv  = tn / (tn + fn) if (tn+fn) else float("nan")
    ci = hanley_mcneil(auroc, int(y_true_bin.sum()), int((y_true_bin == 0).sum()))
    return {"n":int(len(y_true_bin)),"auroc":float(auroc),"auroc_ci95":[ci["low"],ci["high"]],
            "ap":float(ap),"accuracy":acc,"macro_f1":f1m,
            "ppv":float(ppv),"sensitivity":float(sens),"specificity":float(spec),
            "npv":float(npv),"threshold":float(thr),
            "cm":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}}


def prevalence_adjust(sens, spec, prev) -> Dict:
    ppv = (sens*prev) / (sens*prev + (1-spec)*(1-prev)) if (sens*prev + (1-spec)*(1-prev)) else float("nan")
    npv = (spec*(1-prev)) / (spec*(1-prev) + (1-sens)*prev) if (spec*(1-prev) + (1-sens)*prev) else float("nan")
    return {"prevalence":prev, "ppv":float(ppv), "npv":float(npv)}


def plot_roc(y_true_bin, y_score_pos, title, out_path):
    fpr, tpr, _ = roc_curve(y_true_bin, y_score_pos, pos_label=1)
    auroc = roc_auc_score(y_true_bin, y_score_pos)
    plt.figure()
    plt.plot(fpr, tpr, color="tab:blue", label=f"AUROC = {auroc:.3f}")
    plt.plot([0,1],[0,1],"--", color="grey")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def save_predictions(df, y, s, thr, path):
    out = df[["participant_id","label"]].reset_index(drop=True).copy()
    assert np.array_equal(out.label.to_numpy(), y)
    out["p_crc"] = s; out["prediction"] = (s >= thr).astype(int)
    out.to_csv(path, index=False)




def build_scheduler(optimizer, total_steps, warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def main(cfg_path: str):
    C = load_config(cfg_path)
    set_seed(int(C.get("seed", 42)))

    root = Path(C["data"]["root"]).expanduser().resolve()
    out  = ensure_dir(Path(C["output_dir"]).expanduser().resolve())
    figs = ensure_dir(out / "figs"); logs = ensure_dir(out / "logs"); ev = ensure_dir(out / "evaluation")
    class_names = C["data"]["class_names"]; pos_class = C["data"]["pos_class"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_json(environment(), logs/"environment.json")
    save_json(C, out/"config_used.json")


    splits, split_stats = participant_level_split(
        root, class_names, pos_class, C["data"]["split"], int(C.get("seed",42)))
    save_json(split_stats, logs/"split_stats.json")
    print("Split (participants per class):", split_stats)
    manifest_path = out / "manifest.csv"
    manifest = write_manifest(splits, manifest_path)

    train_tf, eval_tf = build_transforms(C)
    mirror = bool(C["training"].get("mirror_views", True))
    train_ds = ListDataset(splits["train"], train_tf, mirror=mirror)
    val_ds   = ListDataset(splits["val"],   eval_tf)
    test_ds  = ListDataset(splits["test"],  eval_tf)


    y_train = np.array([y for _, y, _ in train_ds.items])
    counts = np.bincount(y_train, minlength=len(class_names))
    inv = 1.0 / (counts + 1e-9)
    weights = [float(inv[y]) for y in y_train]
    n_draws = int(round(len(weights) * float(C["training"].get("oversample_factor", 1.0))))
    sampler = WeightedRandomSampler(weights, num_samples=n_draws, replacement=True)
    save_json({"unique_training_images": len(splits["train"]), "training_views": len(train_ds),
               "draws_per_epoch": n_draws}, logs/"sampling.json")

    bs = int(C["training"]["batch_size"]); nw = int(C["training"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    model = build_model(C, pretrained=bool(C["model"]["pretrained"]))
    if C["model"].get("grad_checkpointing", False):
        try: model.set_grad_checkpointing(True)
        except Exception: pass
    model.to(device)


    criterion = nn.CrossEntropyLoss(label_smoothing=float(C["training"]["label_smoothing"]))
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(C["training"]["lr"]),
                                  weight_decay=float(C["training"]["weight_decay"]))
    accum = int(C["training"].get("grad_accum_steps", 1))
    epochs = int(C["training"]["epochs"])
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(C["training"].get("warmup_epochs", 0)) * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(C["training"].get("amp_fp16", True)) and device.type=="cuda")

    patience = int(C["training"]["early_stopping_patience"])
    best_val_loss = float("inf"); wait = 0
    best_ckpt = out / "best.pth"
    history = []

    print(f"Device={device} | backbone={C['model']['backbone']} | "
          f"eff.batch={bs*accum} | epochs={epochs} | draws/epoch={n_draws}")
    for ep in range(1, epochs+1):
        model.train(); t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        run_loss, seen, opt_step = 0.0, 0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                logits = model(x); loss = criterion(logits, y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(); opt_step += 1
            run_loss += loss.item()*accum*x.size(0); seen += x.size(0)
        tr_loss = run_loss / max(1, seen)


        yv, pv, val_loss = collect_scores(model, val_loader, device)
        yv_bin = yv.astype(int); sv = pv[:, class_names.index(pos_class)]
        val_auroc = roc_auc_score(yv_bin, sv)
        history.append({"epoch":ep, "train_loss":tr_loss, "val_loss":val_loss,
                        "val_auroc":float(val_auroc), "lr":optimizer.param_groups[0]["lr"]})
        print(f"Ep {ep:02d}/{epochs} | train_loss={tr_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_AUROC={val_auroc:.3f} | {time.time()-t0:.0f}s")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss; wait = 0
            torch.save({"model": model.state_dict(), "config": C, "epoch": ep,
                        "manifest_sha256": sha256(manifest_path)}, best_ckpt)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep} (best val_loss={best_val_loss:.4f}).")
                break

    save_json(history, logs/"history.json")


    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True)["model"])
    pos = class_names.index(pos_class)

    yv, pv, _ = collect_scores(model, val_loader, device)
    yt, pt, _ = collect_scores(model, test_loader, device)
    ytr, ptr, _ = collect_scores(copy.deepcopy(model), train_loader, device, train_mode=True)
    yv_b, sv = yv.astype(int), pv[:, pos]
    yt_b, st = yt.astype(int), pt[:, pos]
    ytr_b, str_ = ytr.astype(int), ptr[:, pos]

    thr = youden_threshold(yv_b, sv)
    table2 = {
        "training":   metrics_at_threshold(ytr_b, str_, thr),
        "validation": metrics_at_threshold(yv_b, sv, thr),
        "testing":    metrics_at_threshold(yt_b, st, thr),
        "threshold_source": "youden_on_validation",
    }
    save_json(table2, logs/"table_metrics.json")

    save_json({"threshold": thr, "checkpoint_sha256": sha256(best_ckpt),
               "manifest_sha256": sha256(manifest_path)}, ev/"threshold.json")
    save_predictions(manifest[manifest.split=="val"], yv_b, sv, thr, ev/"predictions_val.csv")
    save_predictions(manifest[manifest.split=="test"], yt_b, st, thr, ev/"predictions_test.csv")


    n_pos, n_neg = int(yt_b.sum()), int((yt_b == 0).sum())
    ci = hanley_mcneil(table2["testing"]["auroc"], n_pos, n_neg)
    power = {"n_pos": n_pos, "n_neg": n_neg, "alpha": 0.05,
             "ci95_half_width": 1.96 * ci["se"],
             "power": {f"{a:.2f}": power_auroc(a, n_pos, n_neg) for a in [0.55, 0.60, 0.65, 0.70]}}
    save_json(power, logs/"power.json")


    sens = table2["testing"]["sensitivity"]; spec = table2["testing"]["specificity"]
    prevs = C["evaluation"]["prevalences"]
    table3 = {name: prevalence_adjust(sens, spec, float(p)) for name, p in prevs.items()}
    save_json(table3, logs/"table3_prevalence_adjusted.json")


    plot_roc(ytr_b, str_, "ROC — Training",   figs/"roc_training.png")
    plot_roc(yv_b,  sv,   "ROC — Validation", figs/"roc_validation.png")
    plot_roc(yt_b,  st,   "ROC — Testing",    figs/"roc_testing.png")

    torch.save(model.state_dict(), out/"model_final.pth")


    for k in ["training","validation","testing"]:
        m = table2[k]
        print(f"{k:10s} n={m['n']} AUROC={m['auroc']:.3f} [{m['auroc_ci95'][0]:.3f}-{m['auroc_ci95'][1]:.3f}] "
              f"AP={m['ap']:.3f} Acc={m['accuracy']*100:.2f} "
              f"F1={m['macro_f1']*100:.2f} PPV={m['ppv']*100:.2f} Sens={m['sensitivity']*100:.2f} "
              f"Spec={m['specificity']*100:.2f}")

    for name, v in table3.items():
        print(f"{name:10s} prev={v['prevalence']*100:.2f}% PPV={v['ppv']*100:.2f}% NPV={v['npv']*100:.2f}%")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
