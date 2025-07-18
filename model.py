import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import dgl.nn as dglnn
import pennylane as qml
from pennylane import numpy as np

# ---------- Graph Convolution Network ----------
class GraphConvolutionNetwork(nn.Module):
    def __init__(self, in_dim=74, hidden_dim1=256, hidden_dim2=128):
        super(GraphConvolutionNetwork, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim1)
        self.conv2 = dglnn.GraphConv(hidden_dim1, hidden_dim2)

    def forward(self, graph):
        h = graph.ndata['feat']
        h = F.relu(self.conv1(graph, h))
        h = self.conv2(graph, h)
        with graph.local_scope():
            graph.ndata['h'] = h
            hg = dgl.mean_nodes(graph, 'h')
        return hg

# ---------- Ligand Encoder ----------
class LigandEncoder(nn.Module):
    def __init__(self):
        super(LigandEncoder, self).__init__()
        self.smiles_encoder = AutoModel.from_pretrained("seyonechem/ChemBERTa-zinc-base-v1")
        self.graph_encoder = GraphConvolutionNetwork()

    def forward(self, smiles, graph):
        smiles_out = self.smiles_encoder(**smiles).last_hidden_state.mean(dim=1)
        graph_out = self.graph_encoder(graph)
        return smiles_out, graph_out

# ---------- Protein Encoder ----------
class ProteinEncoder(nn.Module):
    def __init__(self):
        super(ProteinEncoder, self).__init__()
        self.prot_encoder = AutoModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    def forward(self, protein):
        out = self.prot_encoder(**protein).last_hidden_state.mean(dim=1)
        return out

# ---------- Cross-Modal Attention Fusion ----------
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossModalAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, protein, smiles, graph):
        ligand = torch.cat((smiles, graph), dim=1).unsqueeze(1)
        query = protein.unsqueeze(1)
        fused, _ = self.attn(query, ligand, ligand)
        return fused.squeeze(1)

# ---------- Variational Quantum Circuit ----------
class VariationalQuantumCircuit(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super(VariationalQuantumCircuit, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")

    def circuit(self, weights, x):
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        weights = torch.rand(self.n_layers, self.n_qubits, 2, requires_grad=True)
        return self.qnode(weights, x[0])

# ---------- Hybrid Prediction Head ----------
class HybridPredictionHead(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super(HybridPredictionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm(x)
        return self.fc2(x)

# ---------- Full Q-BAFNet Model ----------
class QBAFNet(nn.Module):
    def __init__(self):
        super(QBAFNet, self).__init__()
        self.ligand = LigandEncoder()
        self.protein = ProteinEncoder()
        self.fusion = CrossModalAttention(embed_dim=512)
        self.vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
        self.predictor = HybridPredictionHead(input_dim=4, hidden_dim=64, output_dim=1)

    def forward(self, protein_seq, smiles, graph):
        prot_emb = self.protein(protein_seq)
        smiles_emb, graph_emb = self.ligand(smiles, graph)
        fused = self.fusion(prot_emb, smiles_emb, graph_emb)
        q_embed = self.vqc(fused)
        return self.predictor(q_embed)
