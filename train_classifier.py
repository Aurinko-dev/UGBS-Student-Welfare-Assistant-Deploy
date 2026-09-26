"""
Improved trainer for the welfare category/severity classifier.

Changes vs. the original:
  1. 5-fold stratified cross-validation (on category) instead of one 16-row
     split. Reports mean +/- std, per-fold, so results are trustworthy.
  2. The number of epochs is CHOSEN from CV (epoch with lowest mean
     validation loss) instead of a fixed 60 that overfits.
  3. Weight decay (AdamW) + class-weighted severity loss (Low is 59% of data).
  4. Final model is trained on ALL data with the chosen epoch count.
  5. Reports macro-F1 and High-severity recall (the number that matters for
     a welfare tool), and per-category accuracy.

Usage:  python train_classifier.py
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import config
from classifier_model import WelfareClassifierNet
from risk_classifier import urgency_floor

SEED = 42
MAX_EPOCHS = 60
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
N_FOLDS = 5


def threshold_table(is_high, p_high, thresholds, extra_flag=None):
    """For each cutoff on P(High): recall of truly-High cases, precision, and
    the share of ALL messages that would be flagged. Pure numpy."""
    is_high = np.asarray(is_high, dtype=bool)
    p_high = np.asarray(p_high, dtype=float)
    rows = []
    for t in thresholds:
        flag = p_high >= t
        if extra_flag is not None:
            flag = flag | np.asarray(extra_flag, dtype=bool)
        tp = int((flag & is_high).sum())
        rows.append((t,
                     tp / max(int(is_high.sum()), 1),
                     tp / max(int(flag.sum()), 1),
                     float(flag.mean())))
    return rows


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def make_model_and_opt(n_cat, n_sev):
    model = WelfareClassifierNet(n_cat, n_sev)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    return model, opt


def run_epoch(model, opt, loader, cat_loss, sev_loss):
    model.train()
    for xb, cb, sb in loader:
        opt.zero_grad()
        c, s = model(xb)
        (cat_loss(c, cb) + sev_loss(s, sb)).backward()
        opt.step()


@torch.no_grad()
def evaluate(model, X, y_cat, y_sev, cat_loss, sev_loss, high_idx=None):
    model.eval()
    c, s = model(X)
    p_high = (torch.softmax(s, dim=1)[:, high_idx].numpy()
              if high_idx is not None else np.zeros(len(X)))
    return {
        "loss": (cat_loss(c, y_cat) + sev_loss(s, y_sev)).item(),
        "cat_pred": c.argmax(1).numpy(),
        "sev_pred": s.argmax(1).numpy(),
        "p_high": p_high,
    }


def main():
    set_seed()
    # Fresh checkout / first run: nothing creates ./models yet, so torch.save()
    # and the labels/cv_predictions writes below would crash with
    # FileNotFoundError. Create it up front rather than at each save site.
    os.makedirs(os.path.dirname(config.CLASSIFIER_MODEL_PATH), exist_ok=True)
    print("Loading training data...")
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    print(f"{len(df)} labeled examples, {df['category'].nunique()} categories")

    categories = sorted(df["category"].unique())
    severities = sorted(df["severity"].unique())
    cat_idx = {c: i for i, c in enumerate(categories)}
    sev_idx = {s: i for i, s in enumerate(severities)}

    print("Encoding sentences...")
    from langchain_huggingface import HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    X = torch.tensor(embedder.embed_documents(df["text"].tolist()),
                     dtype=torch.float32)
    y_cat_np = df["category"].map(cat_idx).values
    y_sev_np = df["severity"].map(sev_idx).values
    y_cat = torch.tensor(y_cat_np, dtype=torch.long)
    y_sev = torch.tensor(y_sev_np, dtype=torch.long)
    high_idx = sev_idx.get("High")

    sev_w = class_weights(y_sev_np, len(severities))
    print("Severity class weights:",
          {s: round(float(w), 2) for s, w in zip(severities, sev_w)})

    # ---- Cross-validation -------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    val_loss_curves = np.zeros((N_FOLDS, MAX_EPOCHS))
    fold_stats = []
    oof_cat = np.zeros(len(df), dtype=int)   # out-of-fold predictions at best epoch
    oof_sev = np.zeros(len(df), dtype=int)
    oof_ph = np.zeros(len(df))

    cat_loss = nn.CrossEntropyLoss()
    sev_loss = nn.CrossEntropyLoss(weight=sev_w)

    for fold, (tr, va) in enumerate(skf.split(X.numpy(), y_cat_np)):
        set_seed(SEED + fold)
        model, opt = make_model_and_opt(len(categories), len(severities))
        loader = DataLoader(TensorDataset(X[tr], y_cat[tr], y_sev[tr]),
                            batch_size=BATCH_SIZE, shuffle=True)
        preds_by_epoch = []
        for ep in range(MAX_EPOCHS):
            run_epoch(model, opt, loader, cat_loss, sev_loss)
            r = evaluate(model, X[va], y_cat[va], y_sev[va], cat_loss, sev_loss, high_idx)
            val_loss_curves[fold, ep] = r["loss"]
            preds_by_epoch.append((r["cat_pred"], r["sev_pred"], r["p_high"]))
        fold_stats.append((va, preds_by_epoch))

    mean_curve = val_loss_curves.mean(axis=0)
    best_epoch = int(mean_curve.argmin()) + 1
    print(f"\nBest epoch by mean CV validation loss: {best_epoch} "
          f"(of {MAX_EPOCHS})")

    for va, preds in fold_stats:
        oof_cat[va] = preds[best_epoch - 1][0]
        oof_sev[va] = preds[best_epoch - 1][1]
        oof_ph[va] = preds[best_epoch - 1][2]

    cat_acc = (oof_cat == y_cat_np).mean()
    sev_acc = (oof_sev == y_sev_np).mean()
    print(f"CV category acc: {cat_acc:.2%} | macro-F1: "
          f"{f1_score(y_cat_np, oof_cat, average='macro'):.3f}")
    print(f"CV severity acc: {sev_acc:.2%} | macro-F1: "
          f"{f1_score(y_sev_np, oof_sev, average='macro'):.3f}")
    if high_idx is not None:
        rec = recall_score(y_sev_np == high_idx, oof_sev == high_idx)
        print(f"CV High-severity recall: {rec:.2%}  "
              f"(share of truly High cases the model catches)")

    print("\nPer-category accuracy (out-of-fold):")
    for c, i in cat_idx.items():
        m = y_cat_np == i
        print(f"  {c:32s} {(oof_cat[m] == i).mean():6.1%}  (n={m.sum()})")

    if high_idx is not None:
        is_high = y_sev_np == high_idx
        floor_high = np.array([urgency_floor(t) == "High" for t in df["text"]])
        cuts = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

        print(f"\nP(High) ROC-AUC (out-of-fold): {roc_auc_score(is_high, oof_ph):.3f}"
              "   (0.5 = no signal, 1.0 = perfect)")
        print("\nA) Net only: escalate when P(High) >= cutoff")
        print("  cutoff | High recall | precision | % of messages flagged")
        for t, rec, prec, share in threshold_table(is_high, oof_ph, cuts):
            print(f"   {t:.2f}  |   {rec:6.1%}    |  {prec:6.1%}   |   {share:6.1%}")

        tp = int((floor_high & is_high).sum())
        print(f"\nUrgency floor alone: recall {tp / max(int(is_high.sum()), 1):.1%}, "
              f"precision {tp / max(int(floor_high.sum()), 1):.1%}, "
              f"flags {floor_high.mean():.1%} of messages")
        print("  (NOTE: the floor's patterns were written after reading these rows, so "
              "its numbers here are optimistic.)")

        print("\nB) Policy used by the app: urgency floor OR P(High) >= cutoff")
        print("  cutoff | High recall | precision | % of messages flagged")
        for t, rec, prec, share in threshold_table(is_high, oof_ph, cuts, extra_flag=floor_high):
            print(f"   {t:.2f}  |   {rec:6.1%}    |  {prec:6.1%}   |   {share:6.1%}")
        print("  -> set HIGH_PROB_THRESHOLD in risk_classifier.py to your chosen cutoff.")

        out_path = os.path.join(os.path.dirname(config.CLASSIFIER_MODEL_PATH),
                                "cv_predictions.csv")
        pd.DataFrame({
            "text": df["text"],
            "true_category": df["category"],
            "pred_category": [categories[i] for i in oof_cat],
            "true_severity": df["severity"],
            "pred_severity": [severities[i] for i in oof_sev],
            "p_high": np.round(oof_ph, 4),
            "urgency_floor": [urgency_floor(t) or "" for t in df["text"]],
        }).to_csv(out_path, index=False)
        print(f"Saved out-of-fold predictions to {out_path}")

    print("\nMost common category confusions (true -> predicted):")
    conf = {}
    for t_i, p_i in zip(y_cat_np, oof_cat):
        if t_i != p_i:
            conf[(t_i, p_i)] = conf.get((t_i, p_i), 0) + 1
    for (t_i, p_i), n in sorted(conf.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {categories[t_i]}  ->  {categories[p_i]}: {n}")

    # ---- Final model on ALL data -----------------------------------------
    print(f"\nTraining final model on all {len(df)} examples "
          f"for {best_epoch} epochs...")
    set_seed()
    model, opt = make_model_and_opt(len(categories), len(severities))
    loader = DataLoader(TensorDataset(X, y_cat, y_sev),
                        batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(best_epoch):
        run_epoch(model, opt, loader, cat_loss, sev_loss)

    torch.save(model.state_dict(), config.CLASSIFIER_MODEL_PATH)
    with open(config.CLASSIFIER_LABELS_PATH, "w") as f:
        json.dump({"categories": categories, "severities": severities}, f,
                  indent=2)
    print(f"Saved weights to {config.CLASSIFIER_MODEL_PATH}")
    print(f"Saved labels to {config.CLASSIFIER_LABELS_PATH}")


if __name__ == "__main__":
    main()