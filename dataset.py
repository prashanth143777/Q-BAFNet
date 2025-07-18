import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import AllChem
import dgl
import os

class BindingAffinityDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained("seyonecz/ChemBERTa-zinc-base-v1")
        self.protein_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.scaler = MinMaxScaler()

        # Read and scale affinities
        with open(data_path, 'r') as f:
            lines = f.readlines()
        affinities = [float(line.strip().split()[-1]) for line in lines]
        self.scaler.fit([[a] for a in affinities])

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            identifier, protein_name, smiles, protein_seq, affinity = parts
            graph = self.smiles_to_graph(smiles)
            if graph is None:
                continue
            self.data.append({
                'id': identifier,
                'protein': protein_seq,
                'smiles': smiles,
                'graph': graph,
                'affinity': self.scaler.transform([[float(affinity)]])[0][0]
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        protein_input = self.protein_tokenizer(entry['protein'], return_tensors="pt", padding=True, truncation=True)
        smiles_input = self.tokenizer(entry['smiles'], return_tensors="pt", padding=True, truncation=True)
        graph = entry['graph']
        affinity = torch.tensor(entry['affinity'], dtype=torch.float32)
        return protein_input, smiles_input, graph, affinity

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol)

        g = dgl.DGLGraph()
        num_atoms = mol.GetNumAtoms()
        g.add_nodes(num_atoms)

        node_feats = []
        for atom in mol.GetAtoms():
            node_feats.append([atom.GetAtomicNum()])
        g.ndata['feat'] = torch.tensor(node_feats).float()

        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            g.add_edges(u, v)
            g.add_edges(v, u)
        return g
