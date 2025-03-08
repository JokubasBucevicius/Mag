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

def evaluate_model(batch_size, mode, model_path, device):
    """
    Loads a trained model and evaluates it on the saved test dataset.
    """
    # Load the test dataset associated with this model
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
    model.load_state_dict(torch.load(model_path, weights_only =True))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)

            if mode == "binary":
                preds = (out > 0.3).long()  # Use threshold (same as training)
            else:
                preds = out.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

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

