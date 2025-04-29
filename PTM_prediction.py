#!/usr/bin/env python3
import sys, pandas as pd, h5py, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Configuration
H5_PATH    = "embeddings/embeddings_filtered.h5"
PTM_CSV    = "all_ptms_filtered_F1_isoforms.csv"
INPUT_DIM  = 960
LR         = 2e-4
BATCH_SIZE = 24
EPOCHS     = 5
HIDDEN_DIM = 128
ALPHA      = 10.0
GAMMA      = 2.0
THRESHOLD  = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CONFIG: device={device}, batch={BATCH_SIZE}, epochs={EPOCHS}", flush=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight
    def forward(self, logits, target):
        logpt = -nn.functional.cross_entropy(logits, target,
                                             weight=self.weight,
                                             reduction='none')
        pt = torch.exp(logpt)
        return ((1 - pt) ** self.gamma) * (-logpt)

# Load labels
ptm_df = pd.read_csv(PTM_CSV)
ptm_types = sorted(ptm_df["PTM_Type"].unique())
type2idx = {t: i+1 for i, t in enumerate(ptm_types)}
idx2type = {0: "no-PTM", **{i: t for t, i in type2idx.items()}}
num_classes = len(ptm_types) + 1

ptm_map = {
    uid: {int(r["Modified_Position"]) - 1: type2idx[r["PTM_Type"]]
          for _, r in grp.iterrows()}
    for uid, grp in ptm_df.groupby("UniProt_ID")
}

# Split
with h5py.File(H5_PATH, 'r') as h5f:
    keys = [k for k in h5f.keys() if k in ptm_map]
train_keys, test_keys = train_test_split(keys, test_size=0.2, random_state=42)

# Dataset
class PTMDataset(Dataset):
    def __init__(self, keys):
        self.keys = keys; self.h5f = None
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        if self.h5f is None:
            self.h5f = h5py.File(H5_PATH, 'r')
        uid = self.keys[idx]; emb = self.h5f[uid][()]
        if emb.ndim == 1:
            emb = emb[None, :]
        elif emb.ndim == 2 and emb.shape[0] == INPUT_DIM:
            emb = emb.T
        else:
            emb = emb.reshape(-1, INPUT_DIM)
        L = emb.shape[0]
        labels = np.zeros(L, dtype=int)
        for pos, cls in ptm_map[uid].items():
            if pos < L:
                labels[pos] = cls
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(labels), L

def collate(batch):
    embs, labs, lengths = zip(*batch)
    embs = pad_sequence(embs, True).to(device)
    labs = pad_sequence(labs, True).to(device)
    lengths = torch.tensor(lengths, device=device)
    max_len = labs.size(1)
    mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
    return embs, labs, mask

# Sampler
with h5py.File(H5_PATH, 'r') as h5f:
    weights = [(len(ptm_map[k]) + 1) / (h5f[k].shape[0] + 1) for k in train_keys]
sampler = WeightedRandomSampler(weights, len(weights), True)

train_loader = DataLoader(PTMDataset(train_keys), BATCH_SIZE, sampler=sampler,
                          collate_fn=collate, num_workers=0)
test_loader  = DataLoader(PTMDataset(test_keys), BATCH_SIZE, False,
                          collate_fn=collate, num_workers=0)

# Model
class PTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(INPUT_DIM, INPUT_DIM, 3, padding=1)
        self.fc1  = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2  = nn.Linear(HIDDEN_DIM, num_classes)
    def forward(self, x):
        x = x.to(self.conv.weight.dtype).transpose(1,2)
        x = self.conv(x).transpose(1,2)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = PTMClassifier().to(device)
cw = torch.ones(num_classes, device=device); cw[1:] = ALPHA
criterion = FocalLoss(GAMMA, cw)
optimizer = torch.optim.Adam(model.parameters(), LR)

# Training & evaluation
for epoch in range(1, EPOCHS+1):
    model.train()
    total = 0
    for emb, labs, mask in train_loader:
        optimizer.zero_grad()
        logits = model(emb)
        loss = criterion(logits.view(-1, num_classes), labs.view(-1))
        loss = (loss * mask.view(-1).float()).sum() / mask.sum()
        loss.backward(); optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch} loss={total/len(train_loader):.4f}", flush=True)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for emb, labs, mask in test_loader:
            logits = model(emb)
            probs = torch.softmax(logits, -1)
            conf, cls = probs.max(-1)
            cls = torch.where((cls>0)&(conf>THRESHOLD), cls, torch.zeros_like(cls))
            flat = mask.view(-1)
            trues.append(labs.view(-1)[flat].cpu().numpy())
            preds.append(cls.view(-1)[flat].cpu().numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)

    print(classification_report(y_true, y_pred,
                                target_names=[idx2type[i] for i in range(num_classes)],
                                zero_division=0), flush=True)
    bin_true = y_true>0; bin_pred = y_pred>0
    print("Binary PTM (P, R, F1):",
          precision_score(bin_true, bin_pred, zero_division=0),
          recall_score(bin_true, bin_pred, zero_division=0),
          f1_score(bin_true, bin_pred, zero_division=0), flush=True)

