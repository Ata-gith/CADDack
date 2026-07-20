"""
Example 03: GNN Graph Construction & Forward Pass
===================================================
Demonstrates caddack.gnn.datasets and caddack.gnn.models — converting
SMILES to PyG Data objects and running a single forward pass through
MolecularGCN and MolecularGINE.

Requires: rdkit + torch + torch-geometric
  conda install -c conda-forge rdkit
  pip install torch torch-geometric
"""

import sys

try:
    import torch
    from torch_geometric.loader import DataLoader

    from caddack.gnn.datasets import (
        smiles_to_graph_arrays,
        to_pyg_data,
        build_pyg_dataset,
    )
    from caddack.gnn.models import MolecularGCN, MolecularGINE
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:")
    print("  conda install -c conda-forge rdkit")
    print("  pip install torch torch-geometric")
    sys.exit(1)

smiles_list = [
    "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",    # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",            # paracetamol
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",    # caffeine
]
targets = [5.2, 6.1, 4.8, 3.9]

# ── 1. GraphArrays: framework-agnostic intermediate ──────────────────────────

print("GraphArrays")
graph = smiles_to_graph_arrays(smiles_list[0])
print(f"Aspirin node_features : {len(graph.node_features)} atoms × {len(graph.node_features[0])} features")
print(f"Aspirin edge_index    : {len(graph.edge_index)} directed edges")
print(f"Aspirin edge_features : {len(graph.edge_features)} × {len(graph.edge_features[0])} features")
print(f"Node feature dims     : [atomic_num, degree, formal_charge, aromatic, hybridization, num_H, in_ring]")
print()

# ── 2. Convert to PyG Data ────────────────────────────────────────────────────

print("PyG Data")
pyg = to_pyg_data(graph, y=targets[0])
print(f"x shape       : {pyg.x.shape}")
print(f"edge_index    : {pyg.edge_index.shape}")
print(f"edge_attr     : {pyg.edge_attr.shape}")
print(f"y             : {pyg.y}")
print()

# ── 3. Build a batched dataset ────────────────────────────────────────────────

print("Dataset + DataLoader")
dataset = build_pyg_dataset(smiles_list, targets)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))
print(f"Batch x           : {batch.x.shape}  (all atoms concatenated)")
print(f"Batch edge_index  : {batch.edge_index.shape}")
print(f"Batch batch vector: {batch.batch.tolist()}  (maps atom to molecule index)")
print()

# ── 4. MolecularGCN forward pass ─────────────────────────────────────────────

print("MolecularGCN")
gcn = MolecularGCN.build(
    in_channels=7,
    hidden_channels=64,
    num_layers=3,
    dropout=0.0,
    task="regression",
)
gcn.eval()
with torch.no_grad():
    preds_gcn = gcn(batch)
print(f"Output shape  : {preds_gcn.shape}   (one scalar per molecule)")
print(f"Predictions   : {preds_gcn.tolist()}")
print()

# ── 5. MolecularGINE forward pass ────────────────────────────────────────────

print("MolecularGINE")
gine = MolecularGINE.build(
    in_channels=7,
    edge_dim=7,
    hidden_channels=64,
    num_layers=3,
    dropout=0.0,
    task="regression",
)
gine.eval()
with torch.no_grad():
    preds_gine = gine(batch)
print(f"Output shape  : {preds_gine.shape}")
print(f"Predictions   : {preds_gine.tolist()}")
print()
print("GINE uses bond features (edge_attr) inside message passing,")
print("while GCN ignores them — GINE is generally preferred for molecules.")
