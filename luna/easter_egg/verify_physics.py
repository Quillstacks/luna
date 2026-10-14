#!/usr/bin/env python3
"""
Script 1 of 2: Stage 2 Multi-View Shadow & Physics Verification Suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates multi-view shadow invariance (Delta h), computes Precision@K,
and runs the 4-tier feature ablation across 25 cross-validation folds.

Can be run directly via:
    $ python -m luna.easter_egg.verify_physics
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

def locate_data():
    """Locate benchmark data either bundled in the package or in repo."""
    # 1. Check bundled package data
    bundled_feat = DATA_DIR / "experiment4_feature_matrix.npy"
    bundled_lbl = DATA_DIR / "experiment4_labels.npy"
    bundled_raw = DATA_DIR / "experiment3_raw_shadow_features.csv"
    if bundled_feat.exists() and bundled_lbl.exists() and bundled_raw.exists():
        return bundled_feat, bundled_lbl, bundled_raw

    # 2. Check workspace relative paths
    search_roots = [
        Path.cwd(),
        Path.cwd().parent,
        PACKAGE_DIR.parents[1],
    ]
    for r in search_roots:
        cand_feat = r / "data/king_crater/metrics/experiment4_feature_matrix.npy"
        cand_lbl = r / "data/king_crater/metrics/experiment4_labels.npy"
        cand_raw = r / "data/king_crater/metrics/experiment3_raw_shadow_features.csv"
        if cand_feat.exists() and cand_lbl.exists() and cand_raw.exists():
            return cand_feat, cand_lbl, cand_raw

    raise FileNotFoundError(
        "Could not locate benchmark data files. Please ensure package data is intact."
    )

def run_evaluation(out_dir: str | Path = "results", verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[luna.stage2] Multi-View Shadow & Physics Invariance Verification")
        print("-" * 65)

    feat_path, lbl_path, raw_path = locate_data()
    X = np.load(feat_path) # (186, 5)
    y = np.load(lbl_path)  # (186,)
    df_raw = pd.read_csv(raw_path)

    n_pits = int(np.sum(y == 1))
    n_craters = int(np.sum(y == 0))
    if verbose:
        print(f"Loaded {len(y)} samples: {n_pits} true pits, {n_craters} background crater false alarms.\n")

    # 1. Multi-View Shadow Depth Invariance (Delta h)
    raw_delta_h_pits = df_raw[df_raw["type"] == "Pit"]["delta_h"].values
    raw_delta_h_negs = df_raw[df_raw["type"] == "Crater"]["delta_h"].values

    pass_pits_35 = float(np.mean(raw_delta_h_pits < 0.35) * 100.0)
    pass_pits_20 = float(np.mean(raw_delta_h_pits < 0.20) * 100.0)
    reject_negs_35 = float(np.mean(raw_delta_h_negs >= 0.35) * 100.0)
    reject_negs_20 = float(np.mean(raw_delta_h_negs >= 0.20) * 100.0)

    if verbose:
        print("1. Multi-View Depth Invariance Statistics (Delta h):")
        print(f"   True Pits (N={n_pits}):     Delta h = {np.mean(raw_delta_h_pits):.4f} +/- {np.std(raw_delta_h_pits):.4f}")
        print(f"                               Pass-Rate (Delta h < 0.35): {pass_pits_35:.1f}%")
        print(f"                               Pass-Rate (Delta h < 0.20): {pass_pits_20:.1f}%")
        print(f"   Craters (N={n_craters}):   Delta h = {np.mean(raw_delta_h_negs):.4f} +/- {np.std(raw_delta_h_negs):.4f}")
        print(f"                               Rejection-Rate (Delta h >= 0.35): {reject_negs_35:.1f}%")
        print(f"                               Rejection-Rate (Delta h >= 0.20): {reject_negs_20:.1f}%\n")

    # 2. Precision@K Ranking Evaluation
    s_scores = X[:, 0]
    rank_idx = np.argsort(-s_scores)
    k_values = [10, 20, 30, 50, 69, 100, 150, 186]
    pk_results = []

    if verbose:
        print("2. Precision@K Benchmark (Stage 1 alone vs. Stage 2 Physics Filtered):")
        print(f"   {'K':>3s} | {'Stage 1 Prec':>12s} | {'Stage 1 Rec':>11s} | {'Stage 2 Prec':>12s} | {'Stage 2 Rec':>11s} | {'Boost':>7s}")
        print("   " + "-" * 68)

    for k in k_values:
        top_idx = rank_idx[:k]
        top_y = y[top_idx]
        top_dh = [df_raw.iloc[i]["delta_h"] for i in top_idx]

        p1 = float(np.sum(top_y == 1) / k)
        r1 = float(np.sum(top_y == 1) / float(n_pits))

        passed_mask = np.array([dh < 0.35 for dh in top_dh])
        passed_y = top_y[passed_mask]

        p2 = float(np.sum(passed_y == 1) / len(passed_y)) if len(passed_y) > 0 else 0.0
        r2 = float(np.sum(passed_y == 1) / float(n_pits)) if len(passed_y) > 0 else 0.0
        boost = (p2 - p1) * 100.0

        pk_results.append({
            "K": k,
            "stage1_precision": p1,
            "stage1_recall": r1,
            "stage2_precision": p2,
            "stage2_recall": r2,
            "precision_boost_pct": boost
        })

        if verbose:
            print(f"   {k:3d} | {p1*100.0:11.1f}% | {r1*100.0:10.1f}% | {p2*100.0:11.1f}% | {r2*100.0:10.1f}% | {boost:+6.1f}%")

    if verbose:
        print()

    # 3. 4-Tier Feature Ablation (Repeated Stratified 5-Fold CV, 25 Folds)
    if verbose:
        print("3. 4-Tier Feature Ablation (Repeated Stratified 5-Fold CV, 25 Folds):")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    feature_configs = {
        "Tier A: Retrieval Alone (S_pithos)": [0],
        "Tier B: Retrieval + Form/Contrast": [0, 2, 4],
        "Tier C: Retrieval + Multi-View Physics (S + Delta h)": [0, 1],
        "Tier D: Full Decision Arbiter": [0, 1, 2, 3, 4],
    }
    ablation_results = {}

    for tier_name, cols in feature_configs.items():
        X_sub = X[:, cols]
        precisions, recalls, f1s, pr_aucs = [], [], [], []

        for tr_idx, te_idx in cv.split(X_sub, y):
            X_tr, X_te = X_sub[tr_idx], X_sub[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=42)
            clf.fit(X_tr_s, y_tr)

            y_pred = clf.predict(X_te_s)
            y_prob = clf.predict_proba(X_te_s)[:, 1]

            precisions.append(precision_score(y_te, y_pred, zero_division=0))
            recalls.append(recall_score(y_te, y_pred, zero_division=0))
            f1s.append(f1_score(y_te, y_pred, zero_division=0))

            p_curve, r_curve, _ = precision_recall_curve(y_te, y_prob)
            pr_aucs.append(auc(r_curve, p_curve))

        ablation_results[tier_name] = {
            "precision_mean": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions)),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "pr_auc_mean": float(np.mean(pr_aucs)),
            "pr_auc_std": float(np.std(pr_aucs)),
        }

        if verbose:
            res = ablation_results[tier_name]
            print(f"   {tier_name:<52s} Precision: {res['precision_mean']:.4f} +/- {res['precision_std']:.4f} | "
                  f"Recall: {res['recall_mean']:.4f} | F1: {res['f1_mean']:.4f}")

    metrics = {
        "multi_view_shadow_invariance": {
            "pits_count": n_pits,
            "craters_count": n_craters,
            "pits_delta_h_mean": float(np.mean(raw_delta_h_pits)),
            "pits_delta_h_std": float(np.std(raw_delta_h_pits)),
            "pits_pass_rate_35": pass_pits_35,
            "pits_pass_rate_20": pass_pits_20,
            "craters_delta_h_mean": float(np.mean(raw_delta_h_negs)),
            "craters_delta_h_std": float(np.std(raw_delta_h_negs)),
            "craters_rejection_rate_35": reject_negs_35,
            "craters_rejection_rate_20": reject_negs_20,
        },
        "precision_at_k": pk_results,
        "ablation_tiers": ablation_results,
    }

    out_file = out_dir / "stage2_physics_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"\n[OK] Results saved to: {out_file}")
        print("=" * 76)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="LUNA Easter Egg: Stage 2 Physics Verification")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to save evaluation JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal output")
    args = parser.parse_args()

    run_evaluation(out_dir=args.out_dir, verbose=not args.quiet)

if __name__ == "__main__":
    main()
