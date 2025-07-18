from rdkit import Chem
from rdkit.Chem import AllChem
import dgl
import torch

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    g = dgl.DGLGraph()
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    node_feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    g.ndata['feat'] = torch.tensor(node_feats).unsqueeze(1).float()

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edges(u, v)
        g.add_edges(v, u)

    return g
