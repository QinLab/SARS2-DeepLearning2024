"""
SHAP stability analysis across independent random subsets.

For each VOC, samples sequences three times (seeds 0, 1, 2), computes
aggregate SHAP values, and reports pairwise Jaccard similarity of the
top-k ranked genome positions across runs.

Usage (run from repo root):
    python3 -m gene_shap.shap_stability
    python3 -m gene_shap.shap_stability --num_seq 500 --top_k 100
    python3 -m gene_shap.shap_stability --num_seq 200 --top_k 50 --seeds 0 1 2 3 4

Outputs:
    gene_shap/stability/jaccard_summary.csv   -- per-VOC pairwise Jaccard table
    gene_shap/stability/jaccard_stats.csv     -- mean and min Jaccard per VOC + overall
    gene_shap/stability/top{k}_positions_<VOC>_seed<s>.csv -- top-k positions per run
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from itertools import combinations
from tqdm import trange

import constants.constants as CONST
import one_hot.one_hot as OneHot


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--num_seq", type=int, default=500,
                    help="Sequences per VOC per run (default: 500)")
parser.add_argument("--top_k",   type=int, default=100,
                    help="Number of top positions to compare (default: 100)")
parser.add_argument("--num_bg",  type=int, default=5,
                    help="Background sequences per VOC for DeepExplainer (default: 5)")
parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2],
                    help="Random seeds for subset sampling (default: 0 1 2)")
args = parser.parse_args()

NUM_SEQ = args.num_seq
TOP_K   = args.top_k
NUM_BG  = args.num_bg
SEEDS   = args.seeds

OUT_DIR = os.path.join(CONST.SHAP_DIR.replace("agg_shap_value", "stability"))
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_sequences(df_var, num_seq, rng):
    """Return one-hot features and IDs for a random sample of num_seq rows."""
    sampled = df_var.sample(n=num_seq, random_state=rng)
    features, ids = [], []
    for _, row in sampled.iterrows():
        features.append(np.array(OneHot.one_hot_encode_seq(row["sequence"])))
        ids.append(row["ID"])
    return np.array(features), ids


def aggregate_shap(explainer, features, var_idx):
    """Return per-position summed SHAP values (shape: [seq_len]) for one VOC class."""
    totals = []
    for i in trange(len(features), desc="  SHAP", leave=False):
        sv = explainer.shap_values(features[i:i+1], check_additivity=False)
        pos_shap = np.sum(sv[var_idx], axis=-1).flatten()   # sum over 7 channels
        if np.sum(np.abs(pos_shap)) > 0:
            totals.append(pos_shap)
    return np.sum(np.abs(totals), axis=0) if totals else np.zeros(CONST.SEQ_SIZE)


def top_k_positions(total_shap, k):
    """Return set of 1-indexed positions with the k largest absolute SHAP."""
    idx = np.argsort(total_shap)[::-1][:k]
    return set(idx + 1)   # 1-indexed to match existing CSV convention


def jaccard(set_a, set_b):
    return len(set_a & set_b) / len(set_a | set_b)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Settings: num_seq={NUM_SEQ}, top_k={TOP_K}, seeds={SEEDS}, num_bg={NUM_BG}")

    # Load data
    df_train = pd.read_csv(CONST.TRAIN_DIR)
    df_test  = pd.read_csv(CONST.TEST_DIR)
    df_all   = pd.concat([df_train, df_test], ignore_index=True)
    df_all   = df_all.drop_duplicates(subset=["ID"], keep="first")

    # Load model
    model = tf.keras.models.load_model(CONST.MODEL_SAVE)

    # Build background (same logic as overall_shap.py: 5 seqs per VOC from train)
    print("Building SHAP background...")
    bg_features = []
    for var in CONST.VOC_WHO:
        df_var_train = df_train[df_train["Variant_VOC"] == var].head(NUM_BG)
        for _, row in df_var_train.iterrows():
            bg_features.append(np.array(OneHot.one_hot_encode_seq(row["sequence"])))
    background = np.array(bg_features)
    explainer = shap.DeepExplainer(model, background)

    # Per-VOC stability analysis
    all_rows = []   # for jaccard_summary.csv

    for var in CONST.VOC_WHO:
        print(f"\n── {var} ──")
        var_idx = CONST.VOC_WHO.index(var)
        df_var  = df_all[df_all["Variant_VOC"] == var].reset_index(drop=True)

        if len(df_var) < NUM_SEQ:
            print(f"  WARNING: only {len(df_var)} sequences available; using all.")

        top_sets = {}
        for seed in SEEDS:
            print(f"  Seed {seed} ...")
            n = min(NUM_SEQ, len(df_var))
            feats, ids = encode_sequences(df_var, n, rng=seed)
            total_shap = aggregate_shap(explainer, feats, var_idx)
            top_pos = top_k_positions(total_shap, TOP_K)
            top_sets[seed] = top_pos

            # Save top-k positions for this run
            pos_df = pd.DataFrame(sorted(top_pos), columns=["position_1indexed"])
            pos_df["shap_rank"] = range(1, len(pos_df) + 1)
            out_path = os.path.join(OUT_DIR, f"top{TOP_K}_positions_{var}_seed{seed}.csv")
            pos_df.to_csv(out_path, index=False)

        # Pairwise Jaccard across seeds
        seed_pairs = list(combinations(SEEDS, 2))
        for s1, s2 in seed_pairs:
            j = jaccard(top_sets[s1], top_sets[s2])
            all_rows.append({
                "VOC": var,
                "seed_A": s1,
                "seed_B": s2,
                "jaccard": round(j, 4),
                "intersection": len(top_sets[s1] & top_sets[s2]),
                "union":        len(top_sets[s1] | top_sets[s2]),
            })
            print(f"  Jaccard (seed {s1} vs {s2}): {j:.4f}  "
                  f"(overlap {len(top_sets[s1] & top_sets[s2])}/{TOP_K})")

    # Save full pairwise table
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "jaccard_summary.csv"), index=False)

    # Save per-VOC stats (mean and min Jaccard)
    stats = (summary_df.groupby("VOC")["jaccard"]
             .agg(mean_jaccard="mean", min_jaccard="min")
             .reset_index())
    overall_mean = summary_df["jaccard"].mean()
    overall_min  = summary_df["jaccard"].min()
    overall_row  = pd.DataFrame([{
        "VOC": "Overall",
        "mean_jaccard": round(overall_mean, 4),
        "min_jaccard":  round(overall_min,  4),
    }])
    stats = pd.concat([stats.round(4), overall_row], ignore_index=True)
    stats.to_csv(os.path.join(OUT_DIR, "jaccard_stats.csv"), index=False)

    print("\n── Results ──")
    print(stats.to_string(index=False))
    print(f"\nFiles written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
