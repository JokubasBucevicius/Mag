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

        # Initializint stats for each residue
        self.residue_counts = {i: 0 for i in range(20)}
        self.residue_bindings = {i: 0 for i in range(20)}
        self.binding_probabilities = {}


    def get_pdb_folders(self):
        """Returns a list of available PDB protein folders in the base path."""
        all_items = os.listdir(self.base_path)
        pdb_folders = [f for f in all_items if os.path.isdir(os.path.join(self.base_path, f))]
        pdb_folders = sorted(pdb_folders)
        print(f"Found {len(pdb_folders)} PDB folders, using {len(pdb_folders[:self.batch_size])}")
        pdb_folders = pdb_folders[:self.batch_size]
        print("Loaded PDB IDs:", ", ".join(pdb_folders))
        return pdb_folders


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
            ##Optional (comment if not used)
            edges_df = self.calculate_edge_weights(nodes_df, edges_df)
            
            pyg_graph = self.convert_to_pyg_graph(nodes_df, edges_df)


            protein_graphs.append({
                "pdb_id": pdb_id,
                "nodes": nodes_df,
                "edges": edges_df,
                "pyg_graph": pyg_graph
            })

        
        print(f"Successfully loaded {len(protein_graphs)} protein graphs (batch size = {self.batch_size})")

        # Calculate and log binding probabilities after loading
        self.finalize_binding_probabilities()

        print("Residue binding probabilities (calculated from loaded data):")
        for res_type, prob in self.binding_probabilities.items():
            print(f"Residue {res_type}: {prob:.3f}")
        print("-" * 40)


        return protein_graphs
    
    def load_and_split(self, train_ratio=0.8):
        protein_graphs = self.load_protein_graphs()
        split_idx = int(len(protein_graphs) * train_ratio)
        train_graphs = protein_graphs[:split_idx]
        test_graphs = protein_graphs[split_idx:]
        return train_graphs, test_graphs

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

        nodes_df = nodes_df[columns_to_keep].copy()

        # Keep only the columns that exist (avoids KeyError if some are missing)
        nodes_df = nodes_df[[col for col in columns_to_keep if col in nodes_df.columns]]

        # Residue binding stats
        for _, row in nodes_df.iterrows():
            residue_type = row["residue_type"]
            self.residue_counts[residue_type] += 1
            if row["bsite"] == 1:
                self.residue_bindings[residue_type] += 1


        # Surface detection
        nodes_df["surface_atom"] = (nodes_df["sas_area"] > 0.0).astype(int)

        # Residue grouping function
        def map_residue_to_group(residue_type):
            charged = {0, 7, 11, 18}    # Arg, Lys, His, Asp
            polar = {1, 3, 8, 13, 15}   # Ser, Gln, Asn, Thr, Cys
            hydrophobic = {2, 5, 6, 9, 10, 12, 14, 17}  # Ala, Ile, Leu, Met, Phe, Val, Trp, Pro
            aromatic = {4, 16}          # Tyr, His (His is both charged and aromatic)

            if residue_type in charged:
                return 0  # Charged
            elif residue_type in polar:
                return 1  # Polar
            elif residue_type in hydrophobic:
                return 2  # Hydrophobic
            elif residue_type in aromatic:
                return 3  # Aromatic
            else:
                return 4  # Unknown (should never happen)
            
        nodes_df["residue_group"] = nodes_df["residue_type"].map(map_residue_to_group)

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
    
    def calculate_edge_weights(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates edge weights based on solvent-accessible surface area (sas_area)
        of the two atoms connected by each edge. Adds a column 'sas_area_weight'.
        """

        # Build a lookup table: atom_index -> sas_area
        sas_area_lookup = nodes_df.set_index("atom_index")["sas_area"].to_dict()

        # Compute sas_area_weight for each edge
        edge_sas_weights = [
        (sas_area_lookup[row["atom_index1"]] + sas_area_lookup[row["atom_index2"]]) / 2
        for _, row in edges_df.iterrows()
        ]

        # Add the new column to edges_df
        edges_df = edges_df.copy()
        edges_df["sas_area_weight"] = edge_sas_weights

        return edges_df

    def calculate_binding_probabilities(self, protein_graphs):
        '''
        Calculating binding probabilities for each amino acid this number will be used later to adjusting the weights for each amino acid
        '''
        residue_counts = {i: 0 for i in range(20)}  # 20 residue types
        residue_bindings = {i: 0 for i in range(20)}

        for graph in protein_graphs:
            nodes = graph["nodes"]
            for _, row in nodes.iterrows():
                residue_counts[row.residue_type] += 1
                if row.bsite == 1:
                    residue_bindings[row.residue_type] += 1

        binding_probabilities = {}
        for residue_type, count in residue_counts.items():
            if count > 0:
                binding_probabilities[residue_type] = residue_bindings[residue_type] / count
            else:
                binding_probabilities[residue_type] = 0.0

        return binding_probabilities

    def finalize_binding_probabilities(self):
        for res_type, count in self.residue_counts.items():
            if count > 0:
                self.binding_probabilities[res_type] = self.residue_bindings[res_type] / count
            else:
                self.binding_probabilities[res_type] = 0.0

    def convert_to_pyg_graph(self, nodes_df, edges_df):
        """
        Converts cleaned nodes and edges dataframes into a PyTorch Geometric Data object.
        """
        import torch
        from torch_geometric.data import Data  # type: ignore

        feature_columns = [
            "residue_type", "atom_type",
            "sas_area", "voromqa_score_a", "voromqa_score_r",
            "volume", "radius",
            "center_x", "center_y", "center_z",
            "surface_atom", "residue_group"
        ]

        x = torch.tensor(nodes_df[feature_columns].values, dtype=torch.float)

        # Node labels (y)
        if self.mode == "binary":
            y = torch.tensor(nodes_df["bsite"].values, dtype=torch.long)
        else:  # multiclass
            y = torch.tensor(nodes_df["bind_type"].map(bind_type_mapping).values, dtype=torch.long)  # type: ignore

        # Edge index (nodes connectivity)
        edge_index = torch.tensor(edges_df[["atom_index1", "atom_index2"]].values.T, dtype=torch.long)

        # Edge attributes (area and distance)
        edge_attr = torch.tensor(edges_df[["area", "distance"]].values, dtype=torch.float)

        if "sas_area_weight" in edges_df.columns:
            sas_area_weight = torch.tensor(edges_df["sas_area_weight"].values, dtype=torch.float).view(-1, 1)
            edge_attr = torch.cat([edge_attr, sas_area_weight], dim=1)

        # Node weighting
        node_weights = torch.ones(len(nodes_df), dtype=torch.float)

        # Surface atoms weight
        node_weights[nodes_df["surface_atom"] == 1] *= 1.5

        # Residue types  weight for each amino acid
        for res_type, prob in self.binding_probabilities.items():
            node_weights[nodes_df["residue_type"] == res_type] *= (1.0 + prob * 5.0)  # Scale to amplify effect

        # Create PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, node_weights=node_weights)

        return graph_data



