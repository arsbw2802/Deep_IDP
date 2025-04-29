# 1. Download the latest DisProt JSON  (≈ 6 MB)
mkdir -p ref
curl -L -o ref/disprot.json \
  https://disprot.org/api/v1/disprot

# 2. Extract and embed MoRF segments   (creates data/disprot_proto.h5)
python - <<'PY'
import json, re, h5py, torch, esm, tqdm, pathlib, numpy as np
j = json.load(open("ref/disprot.json"))
segments = []
for entry in j["data"]:
    uid = entry["accession"]
    seq = entry["sequence"]
    for ann in entry["regions"]:
        if ann["annotations"]:
            for a in ann["annotations"]:
                if a["type"] == "MoRF":        # only MoRF regions
                    s, e = ann["start"]-1, ann["end"]    # DisProt is 1-based
                    segments.append((f"{uid}_{s}_{e}", seq[s:e]))

print("MoRF segments:", len(segments))
model, alphabet = esm.pretrained.esm3_t12_650M_UR50D()
bc = alphabet.get_batch_converter(); model.eval().cuda()

embs = []
for i in tqdm.tqdm(range(0, len(segments), 32)):
    batch = segments[i:i+32]
    toks = bc(batch)[2].cuda()
    rep = model(toks, repr_layers=[36])["representations"][36]
    embs.append(rep.mean(1).cpu())
P = torch.cat(embs).numpy()
pathlib.Path("data").mkdir(exist_ok=True)
with h5py.File("data/disprot_proto.h5", "w") as h:
    h.create_dataset("P", data=P, compression="gzip")
print("✔ wrote data/disprot_proto.h5  shape", P.shape)
PY

