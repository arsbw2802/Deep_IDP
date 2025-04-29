import pandas as pd
from tqdm import tqdm

# Step 1: Load available UniProt IDs
with open("available_uniprots.txt", "r") as f:
    available_uniprots = set(line.strip() for line in f if line.strip())
print(f"Loaded {len(available_uniprots)} available UniProt IDs.")

# Step 2: Load BioLiP file locally
biolip_file = "BioLiP.txt"  # <-- adjust if needed
binding_records = []

with open(biolip_file, "r") as f:
    lines = f.readlines()

print(f"Loaded {len(lines)} BioLiP entries.")

# Step 3: Parse each line
for line in tqdm(lines, desc="Parsing BioLiP"):
    parts = line.strip().split("\t")
    if len(parts) < 21:
        continue  # Skip incomplete lines

    pdb_id = parts[0].strip()
    receptor_chain = parts[1].strip()
    ligand_id = parts[4].strip().upper()
    uniprot_id = parts[17].strip()
    binding_site_residues = parts[8].strip()

    # Only DNA or RNA ligands
    if ligand_id not in {"DNA", "RNA"}:
        continue

    # Only UniProt IDs we care about
    if uniprot_id not in available_uniprots:
        continue

    if binding_site_residues and binding_site_residues != "-":
        residues = binding_site_residues.split()
        for res in residues:
            if res:
                binding_records.append({
                    "uniprot_id": uniprot_id,
                    "residue_seq_number": res.strip(),
                    "ligand_type": ligand_id,
                    "pdb_id": pdb_id,
                    "receptor_chain": receptor_chain
                })

# Step 4: Save results
if binding_records:
    df = pd.DataFrame(binding_records)
    df.to_csv("binding_labels.tsv", sep="\t", index=False)
    print(f"Saved {len(df)} binding site records to binding_labels.tsv")
else:
    print("No matching binding residues found.")
