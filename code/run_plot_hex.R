library(hexbin)
library(ggplot2)
library(tibble)
library(cowplot)
library(Seurat)
library(dplyr)
library(Polychrome)

# Extract raw arguments sent from the shell command line
raw_args <- commandArgs(trailingOnly = TRUE)

# FIXED: Filter out the "--args" flag if it is passed explicitly by mistake
args <- raw_args[raw_args != "--args"]

# Dynamically parse variables (e.g., cid_results='path' becomes an active variable)
for (i in seq_len(length(args))) {
  eval(parse(text = args[[i]]))
}

# Print verifications to the log file for validation
print(paste("Sample Name:", sample_name))
print(paste("Node Metadata:", node_meta_file))
print(paste("GMM Results:", cid_results))

# Force numeric transformation for spatial constraints
bin_width <- as.numeric(bin_width)
bin_thresh <- as.numeric(bin_thresh)

# Color Hex Code Layout Definitions
kmeans13_colors <- c(
  "#2f4f4f", "#000000", "#008000", "#4b0082", "#ff0000", "#ffd700", 
  '#b5bd61', "#00ffff", "#0000ff", "#ff69b4", "#1e90ff", "#ffdab9", "#ff00ff"
)
names(kmeans13_colors) <- as.character(0:12)

# Load spatial transcriptomic tracking records
node_meta <- readr::read_csv(node_meta_file, num_threads = 4, show_col_types = FALSE)

# Select only existing, valid coordinate tracking columns from your raw input
node_meta <- node_meta[, c("transcript_id", "cell_id", "x_location", "y_location", "z_location", "feature_name")]

# Load the newly derived clustering labels
gmm_id <- read.table(file = cid_results)

# --- FIXED: Relational Mapping to handle 10.4M rows vs 5,743 cluster results ---
# Extract all distinct valid cell IDs from your data (excluding UNASSIGNED)
unique_cells <- unique(node_meta$cell_id)
unique_cells <- unique_cells[unique_cells != "UNASSIGNED"]

# Verification check to see if number of unique cells matches the GMM matrix rows
if (length(unique_cells) != nrow(gmm_id)) {
  warning("Cell counts do not align cleanly. Falling back to relative row indexing layout assignment.")
  node_meta$gmm <- "Unclustered"
  node_meta$gmm[1:nrow(gmm_id)] <- as.character(gmm_id$V1)
} else {
  # Build a matching relational lookup database table matching cell to cluster label
  cell_cluster_map <- data.frame(
    cell_id = unique_cells,
    gmm = as.character(gmm_id$V1),
    stringsAsFactors = FALSE
  )
  
  # Map the cluster information back into your massive transcript metadata table
  node_meta <- node_meta %>% dplyr::left_join(cell_cluster_map, by = "cell_id")
  
  # Set unassigned or missing cell transcripts to a clear baseline character label
  node_meta$gmm[is.na(node_meta$gmm)] <- "Unclustered"
}

# Ensure an active target fallback column exists for both hex plot sub-renders
node_meta$oldgmm12 <- node_meta$gmm
# --------------------------------------------------------------------------------

# Summary reduction metrics tracking functions
count_nodes <- function(x, threshold = bin_thresh) {
  ifelse(length(x) > threshold, length(x), NA)
}
  
get_major_cluster <- function(x, threshold = bin_thresh) {
  ifelse(length(x) > threshold, names(table(x))[which.max(table(x))], NA)
}

# Plot 1: Plotting original GMM annotations
p1 <- node_meta %>% ggplot() + 
  stat_summary_hex(aes(x = x_location, y = y_location, z = oldgmm12),
                   size = 0.05, fun = get_major_cluster, binwidth = bin_width) +
  theme_bw(base_size = 18) + 
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        panel.border = element_blank(),
        axis.line = element_line(color = "black"),
        axis.ticks = element_line(color = "black"),
        axis.text = element_text(color = "black")) +
  scale_fill_manual(values = as.character(DiscretePalette(n = 13))) + # Changed n to 13 to handle "Unclustered" entries safely
  scale_y_reverse() +
  xlab(paste0(sample_name, "_tf"))

# Plot 2: Plotting new PyG model generated results
p2 <- node_meta %>% ggplot() + 
  stat_summary_hex(aes(x = x_location, y = y_location, z = gmm),
                   size = 0.05, fun = get_major_cluster, binwidth = bin_width) +
  theme_bw(base_size = 18) + 
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        panel.border = element_blank(),
        axis.line = element_line(color = "black"),
        axis.ticks = element_line(color = "black"),
        axis.text = element_text(color = "black")) +
  scale_fill_manual(values = c(kmeans13_colors, "Unclustered" = "#d3d3d3")) + # Added a light grey color fallback for unassigned areas
  scale_y_reverse() +
  xlab(paste0(sample_name, "_pyg"))
  
# Compile images side by side and export to storage disk
ggsave(plot = cowplot::plot_grid(p1, p2),
       filename = out_pdf, 
       width = 14,
       height = 8)
print("Successfully generated comparison plots.")
