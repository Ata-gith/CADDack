from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ATOM_FEAT_DIM = 7  # len(_atom_features) output
BOND_FEAT_DIM = 7  # len(_bond_features) output


@dataclass
class GraphArrays:
    """Framework-agnostic molecular graph representation."""

    node_features: list[list[float]]
    edge_index: list[tuple[int, int]]
    edge_features: list[list[float]]


def _require_rdkit():
    try:
        from rdkit import Chem  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "RDKit is required for molecular graph featurization. "
            "Install with `pip install rdkit` (or `pip install caddack[gnn]`)."
        ) from exc
    return Chem


def _atom_features(atom) -> list[float]:
    """Compact atom-level feature vector from common medicinal chemistry priors."""

    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    aromatic = float(atom.GetIsAromatic())
    hybridization = float(int(atom.GetHybridization()))
    num_h = float(atom.GetTotalNumHs())
    in_ring = float(atom.IsInRing())
    return [
        float(atomic_num),
        float(degree),
        float(formal_charge),
        aromatic,
        hybridization,
        num_h,
        in_ring,
    ]


def _bond_features(bond) -> list[float]:
    Chem = _require_rdkit()
    bt = bond.GetBondType()
    bt_single = float(bt == Chem.rdchem.BondType.SINGLE)
    bt_double = float(bt == Chem.rdchem.BondType.DOUBLE)
    bt_triple = float(bt == Chem.rdchem.BondType.TRIPLE)
    bt_aromatic = float(bt == Chem.rdchem.BondType.AROMATIC)
    conjugated = float(bond.GetIsConjugated())
    in_ring = float(bond.IsInRing())
    stereo = float(int(bond.GetStereo()))
    return [bt_single, bt_double, bt_triple, bt_aromatic, conjugated, in_ring, stereo]


def smiles_to_graph_arrays(smiles: str) -> GraphArrays:
    Chem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    if mol.GetNumAtoms() == 0:
        # e.g. the empty string parses to a valid 0-atom Mol; reject it so
        # build_pyg_dataset's skip_invalid path can drop the row cleanly.
        raise ValueError(f"SMILES has no atoms: {smiles!r}")

    node_features = [_atom_features(atom) for atom in mol.GetAtoms()]

    edge_index: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        edge_index.extend([(i, j), (j, i)])
        edge_features.extend([feat, feat])

    return GraphArrays(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )


def graphs_from_smiles(smiles_list: Sequence[str]) -> list[GraphArrays]:
    return [smiles_to_graph_arrays(s) for s in smiles_list]


def to_pyg_data(graph: GraphArrays, y: float | int | None = None):
    """Convert :class:`GraphArrays` to torch-geometric ``Data`` lazily."""

    try:
        import torch
        from torch_geometric.data import Data  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "PyTorch + torch-geometric are required for GNN training."
        ) from exc

    x = torch.tensor(graph.node_features, dtype=torch.float)
    if graph.edge_index:
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(graph.edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEAT_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data


def build_pyg_dataset(
    smiles: Sequence[str],
    targets: Sequence[float | int],
    skip_invalid: bool = True,
):
    """Build a list of torch-geometric ``Data`` objects from SMILES + targets.

    Real-world datasets (e.g. MoleculeNet) contain a handful of SMILES that
    RDKit cannot parse. When ``skip_invalid`` is True (default) such rows are
    dropped with a warning instead of raising, so a single bad row does not
    abort training. Set ``skip_invalid=False`` to restore strict behaviour.
    """
    if len(smiles) != len(targets):
        raise ValueError("smiles and targets must have equal length")
    dataset = []
    n_skipped = 0
    for s, y in zip(smiles, targets):
        try:
            graph = smiles_to_graph_arrays(s)
        except ValueError:
            if not skip_invalid:
                raise
            n_skipped += 1
            continue
        dataset.append(to_pyg_data(graph, y=y))
    if n_skipped:
        warnings.warn(
            f"build_pyg_dataset skipped {n_skipped} unparseable SMILES "
            f"out of {len(smiles)}.",
            stacklevel=2,
        )
    return dataset
