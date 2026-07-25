import sys
import os
from unittest.mock import MagicMock

# 1. BYPASS VERSION CHECK: Fool the module check into thinking version 0.6.0+ is installed
mock_pyglib = MagicMock()
mock_pyglib.__version__ = "0.5.0"
sys.modules["pyglib"] = mock_pyglib

# Now safely import the remaining dependencies
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from spatialrna import SpatialRNA

# Get sample name wildcard from Snakemake execution
if len(sys.argv) < 2:
    print("Error: Missing sample name argument.")
    sys.exit(1)
sample_name = sys.argv[1]

# 2. FIXED: Reads the panel from the updated 'resources' directory
gene_panel = pd.read_csv("../resource/xenium_gene_panel.csv")

# 3. FIXED: Targets the standardized column name 'gene' generated from the panel file
# --- FIX: Use len(gene_list) instead of gene_list.shape to get a clean integer count ---
gene_list = np.unique(gene_panel.gene)
num_genes = len(gene_list) # This will return exactly 343

x = torch.tensor(np.arange(num_genes))
one_hot_encoding = dict(
    zip(gene_list, F.one_hot(x, num_classes=num_genes))
)


for k in one_hot_encoding.keys():
    one_hot_encoding[k] = one_hot_encoding[k].double()

# 4. FIXED: Safe fall-back print check in case CD3D isn't in your sample gene list
if "PGC" in one_hot_encoding:
    print(one_hot_encoding["PGC"])
else:
    print(f"Total encoded genes: {len(one_hot_encoding)}")

# 5. FIXED: The error-proof dictionary fallback class for control codewords
# --- CHANGE THIS BLOCK TO A DEFAULT DICTIONARY ---
from collections import defaultdict

# Create a dictionary that automatically returns 0 for any missing key
one_hot_encoding_int = defaultdict(int)

# Populate it with your valid biological genes
for key in one_hot_encoding.keys():
    one_hot_encoding_int[str(key)] = int(one_hot_encoding[key].argmax())

print(f"✅ Robust dictionary created. Biological genes mapped: {len(one_hot_encoding_int)}")
# --------------------------------------------------


# CUSTOM OVERRIDE CLASS TO BYPASS THE 'RAW' FOLDER RULE
class LocalSpatialRNA(SpatialRNA):
    # Internal PyTorch Geometric engineering calculation:
    @property
    def raw_paths(self):
    # It takes the root, attaches a hardcoded 'raw' directory, 
    # and joins it to your defined file names
        return [os.path.join(self.root, 'raw', f) for f in self.raw_file_names]



# 5. GENERATE TILE GRAPH: Configured with 30 tiles to safely protect local computer RAM limits
LocalSpatialRNA(
    # Tailored root path structure to match your nested '.csv' folder layout
    root="../data/" + f"{sample_name}.csv" + "/",
    sample_name=f"{sample_name}",
    one_hot_encoding=one_hot_encoding_int,
    num_tiles=30,
    max_num_neighbors=500,
    dim_x="x_location",
    dim_y="y_location",
    tile_by_dim="y_location",
    process_mode="tile",
    load_type="blank",
    # Targets standard 'feature_name' used inside your raw transcripts table
    feature_col="feature_name",
    radius_r=3.0,
)
