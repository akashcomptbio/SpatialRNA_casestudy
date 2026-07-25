# python script to run under local dev-spatialrna conda environment for generating subgraph data per sample

from spatialrna import SpatialRNA
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import sys

# Get sample name wildcard from Snakemake execution
sample_name = sys.argv[1]

# 1. FIXED: Pointing to the plural 'resources' folder
gene_panel = pd.read_csv("../resource/xenium_gene_panel.csv")

# 2. FIXED: Targets the standardized column name 'gene' generated from the panel file
gene_list = np.unique(gene_panel.gene)

x = torch.tensor(np.arange(gene_list.shape[0]))

one_hot_encoding = dict(zip(gene_list, F.one_hot(x, num_classes=gene_list.shape[0])))

for k in one_hot_encoding.keys():
    one_hot_encoding[k] = one_hot_encoding[k].double()

# 3. FIXED: Safe fall-back print check in case CD3D isn't in your sample gene list
if "CD3D" in one_hot_encoding:
    print(one_hot_encoding["PGC"])
else:
    print(f"Total encoded genes: {len(one_hot_encoding)}")

one_hot_encoding_int = dict()
for key in one_hot_encoding.keys():
    one_hot_encoding_int[key] = one_hot_encoding[key].argmax()

# CUSTOM OVERRIDE CLASS TO BYPASS THE 'RAW' FOLDER RULE
class LocalSpatialRNA(SpatialRNA):
    @property
    def raw_file_names(self):
        return [f"../{self.sample_name}.csv"]

# Generate Subgraphs. 
sub_g = LocalSpatialRNA(
    # 4. FIXED: Tailored root path structure to match your nested '.csv' folder layout
    root= "../data/" + f"{sample_name}.csv" + "/",
    sample_name=f"{sample_name}",
    one_hot_encoding=one_hot_encoding_int,
    num_tiles=30,             # 5. FIXED: Must match the same number of tiles (30) used in run_build_graph.py
    radius_r=3.0,
    dim_x = "x_location",
    dim_y = "y_location",
    tile_by_dim="y_location",
    load_type="subgraph",
    feature_col="feature_name",
    process_mode="subgraph",
    subgraph_mode="node_based",
    num_seed_nodes=1000,
    # batch_size is inferred
    num_walks=5,
    num_neighbors=[-1,50],
    force_resample=True,
    subgraph_type="induced",
    shuffle=False
)

print(sub_g[0])
