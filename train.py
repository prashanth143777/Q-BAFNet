import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np

from dataset import BindingAffinityDataset
from model import QBAFNet
from metrics import compute_metrics, print_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data_path, epochs=10, batch_size=32, lr=1e-4):
    # Dataset & DataLoader
    dataset = BindingAffinityDataset(data_path)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model, loss, optimizer
    model = QBAFNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for protein_seq, smiles, graphs, affinities in train_loader:
            protein_seq = {k: v.squeeze(1).to(device) for k, v in protein_seq.items()}
            smiles = {k: v.squeeze(1).to(device) for k, v in smiles.items()}
            graphs = [g.to(device) for g in graphs]
            affinities = affinities.to(device)

            optimizer.zero_grad()
            with autocast():
                predictions = model(protein_seq, smiles, graphs).squeeze()
                loss = criterion(predictions, affinities)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for protein_seq, smiles, graphs, affinities in val_loader:
                protein_seq = {k: v.squeeze(1).to(device) for k, v in protein_seq.items()}
                smiles = {k: v.squeeze(1).to(device) for k, v in smiles.items()}
                graphs = [g.to(device) for g in graphs]
                affinities = affinities.to(device)

                preds = model(protein_seq, smiles, graphs).squeeze()
                loss = criterion(preds, affinities)
                total_val_loss += loss.item()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(affinities.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_preds = dataset.scaler.inverse_transform(np.array(val_preds).reshape(-1, 1)).flatten()
        val_labels = dataset.scaler.inverse_transform(np.array(val_labels).reshape(-1, 1)).flatten()
        metrics = compute_metrics(val_labels, val_preds)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        print_metrics(metrics, prefix="Validation ")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "q_bafnet_best.pth")

    plot_losses(train_losses, val_losses)
    evaluate(model, test_loader, dataset)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

def evaluate(model, test_loader, dataset):
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for protein_seq, smiles, graphs, affinities in test_loader:
            protein_seq = {k: v.squeeze(1).to(device) for k, v in protein_seq.items()}
            smiles = {k: v.squeeze(1).to(device) for k, v in smiles.items()}
            graphs = [g.to(device) for g in graphs]
            affinities = affinities.to(device)

            preds = model(protein_seq, smiles, graphs).squeeze()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(affinities.cpu().numpy())

    test_preds = dataset.scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
    test_labels = dataset.scaler.inverse_transform(np.array(test_labels).reshape(-1, 1)).flatten()

    metrics = compute_metrics(test_labels, test_preds)
    print("\nFinal Test Results:")
    print_metrics(metrics, prefix="Test ")

    plot_scatter(test_labels, test_preds)

def plot_scatter(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel("Actual Affinity")
    plt.ylabel("Predicted Affinity")
    plt.title("Predicted vs Actual (Test Set)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_model("davis.txt")
