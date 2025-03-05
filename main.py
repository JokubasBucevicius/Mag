"""
Main script for training Graph Neural Network
"""

from data_loader import DataLoader

def main():
    # Path to one of the datasets (you can change this to dgDNR, vgDNR, or RNR)
    base_path = "../grafai/graphs/k95/dgDNR/"
    batch_size = 3
    mode = "binary"

    # Initialize the DataLoader with chosen base path and batch size
    loader = DataLoader(base_path=base_path, batch_size=5)

    # Load a batch of protein graphs
    protein_graphs = loader.load_protein_graphs()

    # Example of simple processing/confirmation
    for graph in protein_graphs:
        pdb_id = graph['pdb_id']
        pyg_graph = graph["pyg_graph"]

        print(f"\nProtein: {pdb_id}")
        print(f"  Nodes: {pyg_graph.x.shape[0]}")
        print(f"  Edges: {pyg_graph.edge_index.shape[1]}")
        print(f"  Node features shape: {pyg_graph.x.shape}")
        print(f"  Edge features shape: {pyg_graph.edge_attr.shape}")
        print(f"  Unique labels (y): {pyg_graph.y.unique().tolist()}")
        print("-" * 40)


if __name__ == "__main__":
    main()