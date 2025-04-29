#!/usr/bin/env python3
# morf_unsupervised_simple.py (updated NaN-safe)
# =============================================
# Ultra-fast unsupervised MoRF detection via diagonal Mahalanobis anomaly score on embeddings
# Now sanitizes any NaNs in the embeddings before computing statistics.
# Usage:
#   python morf_unsupervised_simple.py \
#     --embeddings_h5 embeddings_filtered.h5 \
#     --out_scores morf_mahalanobis_scores.npz

import argparse
import h5py
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Diagonal Mahalanobis anomaly on embeddings.")
    parser.add_argument('--embeddings_h5', required=True,
                        help='HDF5 with per-protein embedding arrays [L, D] under each key')
    parser.add_argument('--out_scores', required=True,
                        help='Output .npz mapping protein→anomaly scores per residue')
    args = parser.parse_args()

    # 1) First pass: compute global mean and variance (diagonal) across all residues
    print("🔄 First pass: computing feature means/vars…")
    sums = None
    sumsqs = None
    N = 0
    with h5py.File(args.embeddings_h5, 'r') as f:
        for uid in tqdm(f.keys()):
            arr = f[uid][:]
            # if stored as [1, L, D]
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            # sanitize NaNs
            arr = np.nan_to_num(arr, nan=0.0)
            # arr shape [L, D]
            L, D = arr.shape
            if sums is None:
                sums = np.zeros(D, dtype=np.float64)
                sumsqs = np.zeros(D, dtype=np.float64)
            sums += arr.sum(axis=0)
            sumsqs += (arr**2).sum(axis=0)
            N += L
    if N == 0:
        raise RuntimeError("No residues found in embeddings HDF5.")
    mean = sums / N
    var = sumsqs / N - mean**2
    # avoid zero or negative var
    var[var <= 0] = 1e-6
    print(f"✅ Completed: {N} total residues, feature dim={mean.shape[0]}")

    # 2) Second pass: compute per-residue Mahalanobis (diagonal) score
    print("🔄 Second pass: computing anomaly scores per residue…")
    scores = {}
    with h5py.File(args.embeddings_h5, 'r') as f:
        for uid in tqdm(f.keys()):
            arr = f[uid][:]
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            arr = np.nan_to_num(arr, nan=0.0)
            diff = arr - mean
            # compute (x-mean)^2 / var, sum over features
            score = np.sum((diff*diff) / var, axis=1)
            scores[uid] = score

    # 3) Save
    np.savez_compressed(args.out_scores, **scores)
    print(f"✅ Saved per-residue anomaly scores to {args.out_scores}")

if __name__ == '__main__':
    main()
