# R/03_download_cnv_fixed.R

library(TCGAbiolinks)
library(SummarizedExperiment)

cat("=====================================\n")
cat("Downloading TCGA-BRCA CNV data\n")
cat("Fixing duplicate sample issue\n")
cat("=====================================\n")

# Step 1: Query CNV data
query_cnv <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Copy Number Variation",
  data.type = "Gene Level Copy Number"
)

# Step 2: Download data
cat("Downloading data...\n")
GDCdownload(query_cnv, method="api", files.per.chunk=20)

# Step 3: Handle duplicate samples (CRITICAL FIX)
cat("Handling duplicate samples...\n")

results <- query_cnv$results[[1]]

cat("Total files before deduplication:", nrow(results), "\n")

# Keep only one file per patient
results_unique <- results[!duplicated(results$cases), ]

cat("Files after deduplication:", nrow(results_unique), "\n")

# Replace query with cleaned results
query_cnv$results[[1]] <- results_unique

# Step 4: Prepare data
cat("Preparing CNV matrix...\n")
data_cnv <- GDCprepare(query_cnv)

# Step 5: Extract matrix
cnv_matrix <- assay(data_cnv)

cat("CNV matrix dimensions:", dim(cnv_matrix), "\n")

# Step 6: Save output
dir.create("data/raw", recursive=TRUE, showWarnings=FALSE)

write.csv(as.data.frame(cnv_matrix),
          "data/raw/cnv_raw.csv")

cat("=====================================\n")
cat("CNV data saved successfully!\n")
cat("File: data/raw/cnv_raw.csv\n")
cat("=====================================\n")
