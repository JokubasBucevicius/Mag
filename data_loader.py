'''
Script for loading and cleaning the data (creating PyG datasets)
'''


import os
import pandas as pd




class DataLoader:
    def __init__(self, base_path: str, batch_size: int=5, mode: str = "binary"):
        """
        Initializes the data loader.

        :param base_path (str): Path to the main directory (e.g., 'graphs/k95/dgDNR')
        :param batch_size (int): Number of protein graphs to load at once
        :param mode (str): Classification mode - "binary" (binds to NA or not) or "multiclass" (to which NA binds) (this changes the data cleaning step)
        """
        self.base_path = base_path
        self.batch_size = batch_size
        self.mode = mode.lower()
        assert self.mode in ["binary", "multiclass"], "Mode must be 'binary' or 'multiclass'"



    def get_pdb_folders(self):
        """Returns a list of available PDB protein folders in the base path."""
        all_items = os.listdir(self.base_path)
        pdb_folders = [f for f in all_items if os.path.isdir(os.path.join(self.base_path, f))]
        print(f"Found {len(pdb_folders)} PDB folders, using {len(pdb_folders[:self.batch_size])}")
        return pdb_folders[:self.batch_size]


    def load_protein_graphs(self):
        """
        Loads nodes and edges for a small batch of proteins from the given base path.
        Returns a list of (pdb_id, nodes_df, edges_df) tuples.
        """
        pdb_folders = self.get_pdb_folders()
        protein_graphs = []

        for pdb_id in pdb_folders:
            pdb_path = os.path.join(self.base_path, pdb_id)
            nodes_path = os.path.join(pdb_path, 'graph_nodes.csv')
            edges_path = os.path.join(pdb_path, 'graph_links.csv')

            if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
                print(f"Skipping {pdb_id} (missing files)")
                continue
                
            nodes_df = pd.read_csv(nodes_path)
            nodes_df = self.clean_nodes(nodes_df)
            edges_df = pd.read_csv(edges_path)
            edges_df = self.clean_edges(edges_df)

            pyg_graph = self.convert_to_pyg_graph(nodes_df, edges_df)


            protein_graphs.append({
                "pdb_id": pdb_id,
                "nodes": nodes_df,
                "edges": edges_df,
                "pyg_graph": pyg_graph
            })
        
        print(f"Successfully loaded {len(protein_graphs)} protein graphs (batch size = {self.batch_size})")

        return protein_graphs
    
    def clean_nodes(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """ (Helper function) Cleans the nodes DataFrame to retain only useful columns for GNN. """
    
        base_columns = [
            "atom_index", "residue_index",
            "residue_type", "atom_type",
            "sas_area", "voromqa_score_a", "voromqa_score_r",
            "volume", "radius",
            "center_x", "center_y", "center_z"
        ]

        if self.mode == "binary":
            label_column = ["bsite"]  # Binary classification (binds or not)
        else:  # self.mode == "multiclass"
            label_column = ["ssDNA_bind", "dsDNA_bind", "RNA_bind"]  # Multi-class labels

        # Final columns to keep
        columns_to_keep = base_columns + label_column

        # Keep only the columns that exist (avoids KeyError if some are missing)
        nodes_df = nodes_df[[col for col in columns_to_keep if col in nodes_df.columns]]

        # Ensure labels are properly formatted
        if self.mode == "multiclass":
            nodes_df["bind_type"] = nodes_df[["ssDNA_bind", "dsDNA_bind", "RNA_bind"]].idxmax(axis=1)
            nodes_df = nodes_df.drop(columns=["ssDNA_bind", "dsDNA_bind", "RNA_bind"])

        return nodes_df

    def clean_edges(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """ (Helper function) Cleans the edges DataFrame to retain only useful columns for GNN. """
        columns_to_keep = [
            "atom_index1", "atom_index2",
            "area", "distance"  
        ]

        edges_df = edges_df[[col for col in columns_to_keep if col in edges_df.columns]]

        return edges_df

    def convert_to_pyg_graph(self, nodes_df, edges_df):
        """
        Converts cleaned nodes and edges dataframes into a PyTorch Geometric Data object.
        """
        import torch
        from torch_geometric.data import Data # type: ignore

        feature_columns = [
        "residue_type", "atom_type",
        "sas_area", "voromqa_score_a", "voromqa_score_r",
        "volume", "radius",
        "center_x", "center_y", "center_z"
        ]

        x = torch.tensor(nodes_df[feature_columns].values, dtype=torch.float)

        # Node labels (y)
        if self.mode == "binary":
            y = torch.tensor(nodes_df["bsite"].values, dtype=torch.long)  # Binary classification
        else: # self.mode == "multiclass"
            y = torch.tensor(nodes_df["bind_type"].map(bind_type_mapping).values, dtype=torch.long) # type: ignore

        # Edge index (nodes connectivity)
        edge_index = torch.tensor(edges_df[["atom_index1", "atom_index2"]].values.T, dtype=torch.long)

        # Edge attributes (area and distance)
        edge_attr = torch.tensor(edges_df[["area", "distance"]].values, dtype=torch.float)

        # Create PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        return graph_data


