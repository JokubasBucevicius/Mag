import os
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix, recall_score
import pickle

from model import ProteinGAT
from data_loader import DataLoader

def list_saved_models(directory="trained_models"):
    """
    Lists all saved models in the 'trained_models' folder.
    Returns a list of model filenames.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    models = [f for f in os.listdir(directory) if f.endswith('.pt')]

    if not models:
        print(f"No models found in {directory}/")
    else:
        print("Available models:")
        for idx, model in enumerate(models, start=1):
            print(f"[{idx}] {model}")

    return models

def save_pdb_with_binding_probabilities(original_pdb, output_folder, binding_probs):
    """
    Reads a PDB file, adds predicted binding probabilities for protein atoms,
    and saves a modified PDB file, excluding nucleic acid atoms.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure directory exists
    output_pdb = os.path.join(output_folder, os.path.basename(original_pdb).replace(".pdb", "_predicted.pdb"))

    with open(original_pdb, "r") as infile, open(output_pdb, "w") as outfile:
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                residue_name = line[17:20].strip()  # Extract residue type (e.g., ALA, CYS, etc.)

                # Ignore nucleic acid atoms (DNA/RNA bases)
                if residue_name in {"DA", "DT", "DG", "DC", "A", "U", "G", "C"}:
                    outfile.write(line)  # Keep NA atoms but don't modify them
                    continue

                atom_index = int(line[6:11].strip())  # Extract atom index
                binding_prob = binding_probs.get(atom_index, 0.0)  # Default = 0.0 if missing

                # Append binding probability to the line
                new_line = f"{line[:60]}{binding_prob:6.2f}{line[66:]}\n"
                outfile.write(new_line)
            else:
                outfile.write(line)  # Keep non-ATOM lines unchanged

    print(f"✅ Saved modified PDB: {output_pdb}")




def evaluate_model(batch_size, mode, model_path, device):
    """
    Loads a trained model and evaluates it on the saved test dataset.
    Saves PDBs with predicted binding probabilities for protein atoms only.
    """
    # Load the test dataset
    test_data_path = model_path.replace(".pt", "_test_graphs.pkl")
    if not os.path.exists(test_data_path):
        print(f"Error: No test dataset found for {model_path}")
        return

    with open(test_data_path, "rb") as f:
        test_graphs = pickle.load(f)

    print(f"Loaded test dataset from {test_data_path}")

    pyg_graphs = [g["pyg_graph"] for g in test_graphs]
    test_loader = PyGDataLoader(pyg_graphs, batch_size=batch_size, shuffle=False)

    input_dim = pyg_graphs[0].x.shape[1]
    output_dim = 1 if mode == "binary" else 3

    model = ProteinGAT(input_dim, 64, output_dim, heads=4, mode=mode).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    output_pdb_folder = "predicted_pdbs"
    os.makedirs(output_pdb_folder, exist_ok=True)

    # ✅ Correctly process all test proteins in the batch
    test_graph_index = 0  # Keep track of which test graph we are processing
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)

            if mode == "binary":
                raw_probs = out.cpu().numpy().flatten()  # Extract probabilities
                preds = (out > 0.3).long().cpu().numpy()
            else:
                preds = out.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())

            # ✅ Iterate over all graphs in the batch
            for i in range(len(batch.ptr) - 1):
                pdb_id = test_graphs[test_graph_index]["pdb_id"].split("_chain")[0]  # Adjust for naming
                original_pdb_path = f"../strukturos/k95/dgDNR/{pdb_id}.pdb"

                # ✅ Ensure correct slicing for this protein
                start_idx = batch.ptr[i].item()
                end_idx = batch.ptr[i + 1].item()
                protein_probs = raw_probs[start_idx:end_idx]  # ✅ Slice correctly

                # ✅ Save modified PDB
                save_pdb_with_binding_probabilities(original_pdb_path, output_pdb_folder, binding_probs)
                test_graph_index += 1  # Move to next test graph

    print("✅ All PDBs saved with predicted binding probabilities.")

    # Compute evaluation metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Evaluation Results - {model_path}")
    print(f"Node Accuracy: {acc:.4f}")
    print(f"Node Precision: {precision:.4f}")
    print(f"Node Recall: {recall:.4f}")
    print(f"Node F1-Score: {f1:.4f}")




