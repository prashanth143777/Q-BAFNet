import argparse
import torch
from transformers import AutoTokenizer
from model import QBAFNet
from dataset import smiles_to_graph
import numpy as np
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint):
    model = QBAFNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model

def predict_affinity(smiles, protein_sequence, model, scaler):
    # Tokenizers
    smiles_tok = AutoTokenizer.from_pretrained("seyonechem/ChemBERTa-zinc-base-v1")
    prot_tok = AutoTokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    smiles_input = smiles_tok(smiles, return_tensors='pt').to(device)
    prot_input = prot_tok(protein_sequence, return_tensors='pt').to(device)
    graph = smiles_to_graph(smiles).to(device)

    with torch.no_grad():
        output = model(prot_input, smiles_input, [graph]).squeeze()
        affinity = scaler.inverse_transform(np.array([[output.item()]])).flatten()[0]
        return affinity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, required=True)
    parser.add_argument('--protein', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    scaler = MinMaxScaler()
    scaler.fit(np.array([5.0, 10.0, 15.0]).reshape(-1, 1))  # Dummy fit, use real scaler in production

    model = load_model(args.checkpoint)
    affinity = predict_affinity(args.smiles, args.protein, model, scaler)
    print(f"Predicted Binding Affinity: {affinity:.4f}")
