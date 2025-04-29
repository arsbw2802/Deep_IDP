# supervised_morf_per_protein.py
# ============================
# Per-protein supervised MoRF predictor using existing per-protein embeddings

import argparse
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report


def load_disprot_regions(tsv_path):
    """Load DisProt regions TSV into a dict {UniProt_ID: [(start,end), ...]} using 'acc'."""
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)
    regions = {}
    for _, row in df.iterrows():
        uid = row['acc'].split('-')[0]
        start = int(row['start'])
        end   = int(row['end'])
        regions.setdefault(uid, []).append((start, end))
    return regions


def label_site(uid, pos, regions_map):
    """Return 1 if pos in any region for uid, else 0."""
    segs = regions_map.get(uid, [])
    return int(any(s <= pos <= e for s, e in segs))


class MorfNet(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(args):
    # 1) Load per-protein embeddings
    with h5py.File(args.embeddings_h5, 'r') as hf:
        emb_map = {key: hf[key][:] for key in hf.keys()}
    if not emb_map:
        raise RuntimeError(f"No embeddings found in {args.embeddings_h5}")
    print(f"Loaded embeddings for proteins: {list(emb_map.keys())[:5]}... (+{len(emb_map)-5} more)")

    # Determine expected embedding dimension
    first_key = next(iter(emb_map))
    first_emb = emb_map[first_key]
    # if saved per-residue windows, emb_map entries are 2D [L, D]
    # if saved per-protein mean, emb_map entries are 1D [D]
    if first_emb.ndim == 2:
        emb_dim = first_emb.shape[1]
    elif first_emb.ndim == 1:
        emb_dim = first_emb.shape[0]
    else:
        raise RuntimeError(f"Unexpected embedding array shape {first_emb.shape}")
    print(f"Using embedding dimension D={emb_dim}")

    # 2) Load CSV PTM data
    df = pd.read_csv(args.seq_csv)
    df['uid'] = df['UniProt_ID'].str.split('-').str[0]
    df['pos'] = df['Modified_Position'].astype(int)

    # 3) Build training examples: ensure consistent shape
    X_list, y_list = [], []
    skipped = 0
    for _, row in df.iterrows():
        uid, pos = row['uid'], row['pos']
        emb_arr = emb_map.get(uid)
        if emb_arr is None:
            skipped += 1
            continue
        # extract vector at residue pos
        if emb_arr.ndim == 2:
            if pos < 1 or pos > emb_arr.shape[0]:
                skipped += 1
                continue
            vec = emb_arr[pos-1]
        else:
            vec = emb_arr
        if vec.shape != (emb_dim,):
            print(f"Skipping {uid}:{pos} due to shape {vec.shape}")
            skipped += 1
            continue
        X_list.append(vec)
        y_list.append(label_site(uid, pos, regions))
    if not X_list:
        raise RuntimeError("No valid embeddings found for any PTM sites.")
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    print(f"Training on {len(y)} sites, skipped {skipped} invalid rows.")

    # 4) DataLoader
    device = args.device
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # 5) Model setup
    model = MorfNet(emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    # 6) Train
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch}, Loss {total_loss/len(ds):.4f}")

    # 7) Save
    torch.save(model.state_dict(), args.model_out)
    print(f"Saved model to {args.model_out}")

    # 8) Evaluate
    model.eval()
    with torch.no_grad():
        preds = (model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy() > 0.5)
    print(classification_report(y, preds, digits=4))

def predict_model(args):
    # Similar to train, but only build X and run model
    with h5py.File(args.embeddings_h5, 'r') as hf:
        emb_map = {key: hf[key][:] for key in hf.keys()}
    df = pd.read_csv(args.seq_csv)
    df['uid'] = df['UniProt_ID'].str.split('-').str[0]
    df['pos'] = df['Modified_Position'].astype(int)

    X_list, idxs = [], []
    for i, row in df.iterrows():
        uid, pos = row['uid'], row['pos']
        emb = emb_map.get(uid)
        if emb is None or pos<1 or pos>emb.shape[0]:
            continue
        X_list.append(emb[pos-1])
        idxs.append(i)
    X = np.stack(X_list, axis=0)

    device = args.device
    model = MorfNet(X.shape[1]).to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

    out = pd.DataFrame({'index': idxs, 'morf_prob': probs})
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote predictions to {args.out_csv}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    p1 = sub.add_parser('train')
    p1.add_argument('--embeddings_h5', required=True)
    p1.add_argument('--seq_csv',       required=True)
    p1.add_argument('--disprot_tsv',   required=True)
    p1.add_argument('--model_out',     required=True)
    p1.add_argument('--epochs',        type=int,   default=5)
    p1.add_argument('--batch_size',    type=int,   default=512)
    p1.add_argument('--lr',            type=float, default=1e-3)
    p1.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    p2 = sub.add_parser('predict')
    p2.add_argument('--embeddings_h5', required=True)
    p2.add_argument('--seq_csv',       required=True)
    p2.add_argument('--model_ckpt',    required=True)
    p2.add_argument('--out_csv',       required=True)
    p2.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # load DisProt regions once
    global regions
    regions = load_disprot_regions(args.disprot_tsv)
    if args.cmd == 'train': train_model(args)
    elif args.cmd == 'predict': predict_model(args)
    else: parser.print_help()

if __name__=='__main__': main()
