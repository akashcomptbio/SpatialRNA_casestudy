
import spatialrna
import csv
# %%
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import sys 

out_model = sys.argv[1]
epoch = sys.argv[2]
epoch = int(epoch)
model_select = sys.argv[3]

SpatialRNA = spatialrna.SpatialRNA
gene_panel = pd.read_csv("../resource/xenium_gene_panel.csv")
gene_panel


# %%
# Read the 'gene' column without pandas
with open("/home/akash_gro/Spatial_RNA/resource/xenium_gene_panel.csv", mode="r", encoding="utf-8") as f:
    genes_raw = [row["gene"].strip() for row in csv.DictReader(f) if row.get("gene")]
gene_list = np.unique(genes_raw)


x = torch.tensor(np.arange(gene_list.shape[0]))

one_hot_encoding = dict(zip(gene_list, F.one_hot(x, num_classes=gene_list.shape[0])))

for k in one_hot_encoding.keys():
    one_hot_encoding[k] = one_hot_encoding[k].double()
print(one_hot_encoding["CD3D"])


# %%
len(one_hot_encoding)

# %%
one_hot_encoding["CD3E"].argmax()

# %%


# %%
one_hot_encoding_int = dict()
for key in one_hot_encoding.keys():
    one_hot_encoding_int[key] = one_hot_encoding[key].argmax()

# %%
one_hot_encoding_int["CD3E"]

sample_names = []
with open("/home/akash_gro/Spatial_RNA/resource/sample_name.txt", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("new_names"):
            sample_names.append(row["new_names"].strip())
print(f"Loaded {len(sample_names)} active sample IDs from resource mapping index.")

# %%
from torch_geometric.data import Batch, Data

subgraph_list = []
for sample_name in sample_names:
    # Ensure sample_folder has the exact .csv suffix if it's not already there
    sample_folder = sample_name if sample_name.endswith(".csv") else f"{sample_name}.csv"
    
    # The inner file name must NOT contain the .csv suffix
    clean_file_base = sample_folder.replace(".csv", "")
    
    # Construct the exact path matching your directory tree layout
    path = f"/home/akash_gro/Spatial_RNA/data/{sample_folder}/subgraph/{clean_file_base}_subgraph_data_tile0.pt"
    subgraph_list.append(path)

# Load graph data tensors securely using absolute system path parameters
data_list = [Data(**(torch.load(data_path, weights_only=True)[0])) for data_path in subgraph_list]

# %%
len(data_list)

# %%
data_list[0]
d_batch = Batch.from_data_list(data_list)

# --- ADD THIS LOGGING / SAFETY CHECK ---
print(f"Graph batch loaded. Minimum gene ID: {d_batch.x.min().item()}, Maximum gene ID: {d_batch.x.max().item()}")
num_gene_classes = max(int(d_batch.x.max().item()) + 1, len(gene_list), 343)
print(f"Setting dynamic one-hot class space size to: {num_gene_classes}")
d_batch = Batch.from_data_list(data_list)

d_batch.x.shape


# %%
d_batch

# %%
import random as rn
rn.seed(1024)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())
print(device)


from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn import GraphSAGE, GAT, GATConv, AttentionalAggregation
from torch_geometric.nn import GATConv

class GATWithL2Normalization(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, dropout=0.0):
        super(GATWithL2Normalization, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.normalize(x, p=2, dim=1)  # L2-normalize the embeddings

        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=1)  # L2-normalize again after the second layer

        return x


if model_select == "GAT":
    model = GAT(
         343,
         hidden_channels=50,
         num_layers=2,
         v2 = False,
     ).to(device)
elif model_select == "GATv2":
    model = GAT(
         343,
         hidden_channels=50,
         num_layers=2,
         v2 = True,
     ).to(device)
elif model_select == "GATL2":
    model = GATWithL2Normalization(
        in_channels = 343,
        hidden_channels=50,
        out_channels = 50
    ).to(device)
else:
    model = GraphSAGE(
         343,
         hidden_channels=50,
         num_layers=2,
     ).to(device)

print(model)
print("d_batch.x.size() ", d_batch.x.size())
print("d_batch.edge_index.size() ", d_batch.edge_index.size())

print("d_batch.edge_label_index.size() ", d_batch.edge_label_index.size())

from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

train_loader = LinkNeighborLoader(
    d_batch,
    batch_size=2048,
    shuffle=True,
    edge_label_index = d_batch.edge_label_index,
    neg_sampling=NegativeSampling(mode="triplet",amount=1),
    # weight breaks sampling due to large number of nodes.
    #neg_sampling=NegativeSampling(mode="triplet",amount=1,dst_weight=degree(d_batch.edge_index[0])**0.75),
    num_neighbors=[20,10],
    disjoint = False,
    subgraph_type = "bidirectional" ,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train():
    model.train()

    total_loss = 0
    batch_accuracy = []
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        #h = model(F.one_hot(batch.x, num_classes=gene_list.shape[0]).type(torch.float), batch.edge_index)
        b_edge_label_index = torch.concat([torch.stack([batch.src_index, batch.dst_pos_index]),
                                           torch.stack([batch.src_index, batch.dst_neg_index])],1)
        b_edge_label = torch.concat([torch.tensor([1]*batch.src_index.shape[0]),torch.tensor([0]*batch.src_index.shape[0])])

        # --- REPLACE WITH THIS FIXED ONE-HOT DIMENSION ALIGNMENT ---
        x_1d = batch.x.flatten().long()
        
        valid_mask = (x_1d >= 0) & (x_1d < 343)
        safe_x = torch.where(valid_mask, x_1d, torch.zeros_like(x_1d))
        
        oh_matrix = F.one_hot(safe_x, num_classes=343).float()
        batch.x = oh_matrix * valid_mask.float().unsqueeze(-1)
        # ----------------------------------------------------------

        h = model(batch.x, batch.edge_index)
        h_src = h[b_edge_label_index[0]]
        h_dst = h[b_edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred.to(device), b_edge_label.float().to(device))
        # Step 1: Apply Sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(pred)
        # Step 2: Threshold probabilities to get binary predictions
        predictions = (probabilities >= 0.5).float()

        # Step 3: Calculate accuracy
        correct_predictions = (predictions == b_edge_label.float().to(device)).float()
        batch_accuracy = [correct_predictions.sum().cpu() / pred.size(0)] + batch_accuracy
        loss.backward()

        optimizer.step()
        total_loss += float(loss) * pred.size(0)
    return total_loss / d_batch.num_nodes, np.mean(batch_accuracy)




times = []
train_loss = []
train_acc_list = []
## need large epochs, 50.
for epoch in range(1, epoch+1):
    start = time.time()
    loss,acc = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train_Acc: {acc:.2f}')
    times.append(time.time() - start)
    train_loss = train_loss + [loss]
    train_acc_list = train_acc_list + [acc]
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


torch.save(model,out_model)



