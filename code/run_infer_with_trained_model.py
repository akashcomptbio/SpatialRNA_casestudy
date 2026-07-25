import sys
import os.path as osp
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv

# Define the custom architecture exactly as it was during training
class GATWithL2Normalization(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, dropout=0.0):
        super(GATWithL2Normalization, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        return x

# Extract arguments sent by Snakemake
sname = sys.argv[1]
data_path = sys.argv[2]
model_path = sys.argv[3]
out_embs_npy = sys.argv[4]

# Assign GPU device dynamically based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
print(f"Executing inference for sample {sname} on device: {device}")

# Initialize model architecture
model = GATWithL2Normalization(
   in_channels=343,
   hidden_channels=50,
   out_channels=50
)

# FIXED: Safely load model to target device to prevent CPU-GPU mismatch runtimes
# FIXED: Added weights_only=False to allow PyTorch to unpickle your custom PyG model architectures safely
try:
    # Try loading as a state_dict first (Best practice)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
except Exception:
    # Fallback if the entire model object was saved
    model = torch.load(model_path, map_location=device, weights_only=False)

model = model.to(device)
model.eval()

@torch.no_grad()
def inference(subgraph_loader):
    xs = []
    for batch in subgraph_loader:
        # Move execution batch to target hardware (GPU)
        batch = batch.to(device)
        
        # Ensure indices are strictly bounded between 0 and 342
        batch.x = torch.clamp(batch.x, min=0, max=342)
        
        # Reshape and cast to long integer type safely
        batch.x = batch.x.long().squeeze(1) if batch.x.dim() > 1 else batch.x.long()

        # One-hot encode features on the active device
        batch.x = torch.nn.functional.one_hot(batch.x, num_classes=343).float()

        # Predict embeddings
        out = model(batch.x, batch.edge_index)
        out = out[:batch.batch_size]
        xs.append(out.cpu())
        
    x_all = torch.cat(xs, dim=0)
    return x_all

# Securely load the saved geometric tracking data
# Securely load the saved geometric tracking data
loaded_data = torch.load(data_path, map_location='cpu')

# FIXED: Handle nested tuples or lists wrapping the PyG Data object
if isinstance(loaded_data, (tuple, list)):
    print(f"Unboxing data payload from container type: {type(loaded_data)}")
    data_payload = loaded_data[0]
else:
    data_payload = loaded_data

# FIXED: Handle if the inner payload is still a tuple/list, or a dict, or already a Data object
if isinstance(data_payload, (tuple, list)):
    data_payload = data_payload[0]

if isinstance(data_payload, dict):
    data = Data(**data_payload)
elif hasattr(data_payload, 'edge_index'):
    data = data_payload
else:
    raise TypeError(f"Could not convert data payload type {type(data_payload)} into a PyG Data object.")

print(f"Successfully verified object type entering NeighborLoader: {type(data)}")

# Initialize structural sub-graph neighborhood batch loaders
subgraph_loader = NeighborLoader(
    data,
    input_nodes=None,
    num_neighbors=[20, 10],
    batch_size=1024,
    replace=False,
    shuffle=False,
    subgraph_type="induced"
)

# Extract and save spatial node embeddings
node_embs = inference(subgraph_loader).numpy()
np.save(out_embs_npy, node_embs)
print(f"Successfully saved embeddings to {out_embs_npy}")
