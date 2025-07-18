Q-BAFNet: Quantum-Classical Hybrid Model for Drug-Target Affinity Prediction


ChemBERTa and GCNs for ligand representation
ProtT5 for protein sequence encoding
Cross-Modal Attention Fusion for ligand-protein alignment
Variational Quantum Circuit (VQC) to project fused embeddings into a high-dimensional quantum Hilbert space
Hybrid prediction head for final affinity estimation
The datasets are freely available. Due to copyright issues, we do not upload the datasets

Q-BAFNet/
│
├── model.py             # Model definitions: ChemBERTa, ProtT5, GCN, CMAF, VQC  
├── dataset.py           # Dataset loading and preprocessing  
├── train.py             # Main training, validation, testing loop  
├── metrics.py           # Metric computation: MSE, PCC, CI, R2  
├── inference.py         # Inference script for single ligand-protein pair  
├── utils.py             # Helper functions (tokenization, graph creation, etc.)  
├── requirements.txt     # Dependency list  
├── README.md            # Project documentation  

