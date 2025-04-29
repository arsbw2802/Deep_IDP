# eval_morf.py
# =========================
# Evaluate per-residue MoRF anomaly scores against DisProt ground truth,
# robust to any NaNs in the score arrays.
# Usage:
#   python eval_morf.py \
#     --scores morf_mahalanobis_scores.npz \
#     --disprot ref/regions2.tsv \
#     --threshold_percentile 95

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def load_disprot_mask(tsv_path, uid, length):
    df = pd.read_csv(tsv_path, sep='\t', dtype={'acc':str,'start':int,'end':int})
    base = uid.split('-')[0]
    mask = np.zeros(length, dtype=bool)
    for _, row in df[df['acc']==base].iterrows():
        s,e = row['start'], row['end']
        if s<=e and s>=1:
            mask[s-1: min(e, length)] = True
    return mask


def mask_to_regions(mask):
    regions=[]
    start=None
    for i,m in enumerate(mask, start=1):
        if m and start is None:
            start=i
        if not m and start is not None:
            regions.append((start, i-1))
            start=None
    if start is not None:
        regions.append((start,len(mask)))
    return regions


def main():
    p = argparse.ArgumentParser(description="Evaluate MoRF anomaly scores vs DisProt truth.")
    p.add_argument('--scores', required=True)
    p.add_argument('--disprot', required=True)
    p.add_argument('--threshold_percentile', type=float, default=95)
    p.add_argument('--out_pred_regions', default='predicted_morf_regions.csv')
    args = p.parse_args()

    data = np.load(args.scores)
    # gather all valid scores
    all_scores_list = []
    for uid in data.files:
        arr = data[uid]
        # filter out NaNs
        arr = arr[~np.isnan(arr)]
        if arr.size>0:
            all_scores_list.append(arr)
    if not all_scores_list:
        raise RuntimeError("No valid scores found in NPZ.")
    all_scores = np.hstack(all_scores_list)

    # compute threshold
    thresh = np.percentile(all_scores, args.threshold_percentile)
    print(f"Using threshold = {args.threshold_percentile} percentile -> score >= {thresh:.6f}")

    # evaluate per-residue
    y_true_global = []
    y_score_global = []
    y_pred_global = []
    rows = []
    for uid in data.files:
        scores = data[uid]
        # skip entirely NaN
        if np.all(np.isnan(scores)):
            continue
        # fill NaN with very low anomaly (so predicted negative)
        scores_clean = np.nan_to_num(scores, nan=-np.inf)
        mask_gt = load_disprot_mask(args.disprot, uid, len(scores_clean))
        # extend global lists
        y_true_global.append(mask_gt)
        y_score_global.append(scores_clean)
        y_pred_global.append(scores_clean >= thresh)
        # generate region calls
        for (s,e) in mask_to_regions(scores_clean >= thresh):
            rows.append({'UniProt_ID':uid, 'start':s, 'end':e})

    # flatten
    y_true = np.hstack(y_true_global)
    y_score = np.hstack(y_score_global)
    y_pred = np.hstack(y_pred_global)

    # compute metrics
    auc = roc_auc_score(y_true, y_score)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"ROC AUC: {auc:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # save predicted regions
    pd.DataFrame(rows).to_csv(args.out_pred_regions, index=False)
    print(f"Wrote predictions to {args.out_pred_regions}")

if __name__=='__main__':
    main()
