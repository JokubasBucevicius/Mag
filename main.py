"""
Main script for training Graph Neural Network
"""

from data_loader import DataLoader
from train import train_model
from evaluate import evaluate_model, list_saved_models
import torch
import os
import numpy as np

def main():
    # Path to one of the datasets (you can change this to dgDNR, vgDNR, or RNR)
    base_path = "../grafai/graphs/k95/dgDNR/"
    batch_size = 6
    mode = "binary"
    num_epochs = 50
    learning_rate = 0.00001
    hidden_dim = 64
    heads = 4
    threshold = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load and split the data
    loader = DataLoader(base_path, batch_size, mode)
    train_graphs, test_graphs = loader.load_and_split(train_ratio=5/6)

    # Check label and surface imbalance upfront
    print("Checking label and surface imbalance before training starts...")

    all_labels = np.concatenate([g["pyg_graph"].y.cpu().numpy() for g in train_graphs])
    all_surface = np.concatenate([g["nodes"]["surface_atom"].values for g in train_graphs])

    num_binding = np.sum(all_labels)
    num_surface = np.sum(all_surface)
    num_total = len(all_labels)

    binding_percentage = (num_binding / num_total) * 100
    surface_percentage = (num_surface / num_total) * 100
    surface_binding_percentage = (num_binding/num_surface) * 100
    binding_probs = loader.calculate_binding_probabilities(train_graphs)

    print(f"Total Training Nodes: {num_total}")
    print(f"Surface Nodes: {num_surface} ({surface_percentage:.2f}%)")
    print(f"Binding Nodes: {num_binding} ({binding_percentage:.2f}%)")
    print(f"Surface Nodes/Binding Nodes: ({surface_binding_percentage:.2f}%)")
    print("Residue binding probabilities (calculated from training data):")
    for res_type, prob in binding_probs.items():
        print(f"Residue {res_type}: {prob:.3f}")

    print("-" * 40)


    action = input("Choose action: [train/evaluate] ").strip().lower()

    if action == "train":
        model_name = input("Enter model name(e.g., GAT_dgDNR_binary_lr00001_hidden64): ").strip()
        train_model(train_graphs, batch_size, mode, num_epochs, learning_rate, hidden_dim, heads, device, model_name, threshold)

    elif action == "evaluate":
        models = list_saved_models()
        if not models:
            print("No saved models to evaluate. Please train a model first.")
            return
        
        choice = int(input("Select model (enter number): ").strip())
        if choice < 1 or choice > len(models):
            print("Invalid selection.")
            return

        model_name = models[choice - 1]
        model_path = f"trained_models/{model_name}"
        evaluate_model(test_graphs, batch_size, mode, model_path, device)

    else:
        print("Invalid action. Please choose 'train' or 'evaluate'.")


if __name__ == "__main__":
    main()
