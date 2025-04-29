import pandas as pd
import requests
from tqdm import tqdm

# Step 1: Load the binding labels
binding_df = pd.read_csv("binding_labels.tsv", sep="\t")

# Step 2: Group by UniProt ID
binding_group = binding_df.groupby("uniprot_id")["residue_seq_number"].apply(list).to_dict()

# Step 3: Function to fetch UniProt sequence
def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.split("\n")
        sequence = ''.join(lines[1:]) 
        return sequence
    else:
        print(f"Warning: Failed to fetch sequence for {uniprot_id}")
        return None

# Step 4: Build per-residue binding labels
full_records = []

for uniprot_id, residue_list in tqdm(binding_group.items(), desc="Building per-residue labels"):
    sequence = fetch_uniprot_sequence(uniprot_id)
    if sequence is None:
        continue  # skip if can't fetch

    labels = [0] * len(sequence)
    
    for res in residue_list:
        try:
            idx = int(res[1:]) - 1  
            if 0 <= idx < len(sequence):
                labels[idx] = 1
        except ValueError:
            continue  

    full_records.append({
        "uniprot_id": uniprot_id,
        "sequence": sequence,
        "binding_labels": ''.join(map(str, labels)) 
    })

# Step 5: Save
full_df = pd.DataFrame(full_records)
full_df.to_csv("full_per_residue_labels.tsv", sep="\t", index=False)

print(f"Saved {len(full_df)} proteins with full-length binding labels to full_per_residue_labels.tsv")

