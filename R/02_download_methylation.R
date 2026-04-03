# R/02_download_methylation_FIXED.R
library(TCGAbiolinks)
library(SummarizedExperiment)

cat("=====================================\n")
cat("TCGA-BRCA Methylation FIXED\n")
cat("=====================================\n")

# Step 1: Query
query_meth <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "DNA Methylation",
  data.type = "Methylation Beta Value",
  platform = "Illumina Human Methylation 450"
)

# Step 2: Download (skips if already done)
GDCdownload(query_meth, method="api",
            files.per.chunk=5)

# Step 3: Remove duplicates same way as CNV
results <- query_meth$results[[1]]
cat("Total files:", nrow(results), "\n")

# Keep only tumor samples (sample type 01)
barcodes    <- results$cases
sample_type <- substr(barcodes, 14, 15)
results_tumor <- results[sample_type == "01", ]
cat("Tumor samples:", nrow(results_tumor), "\n")

# Remove duplicate patients
patient_id  <- substr(results_tumor$cases, 1, 12)
keep_idx    <- !duplicated(patient_id)
results_clean <- results_tumor[keep_idx, ]
cat("After deduplication:", nrow(results_clean), "\n")

# Replace query results
query_meth$results[[1]] <- results_clean

# Step 4: Prepare
cat("Preparing methylation matrix...\n")
cat("This takes 30-60 minutes...\n")
data_meth <- GDCprepare(query_meth)

# Step 5: Extract matrix
meth_matrix <- assay(data_meth)
cat("Methylation matrix:", dim(meth_matrix), "\n")

# Step 6: Filter probes with >20% missing
missing_pct   <- rowMeans(is.na(meth_matrix))
meth_filtered <- meth_matrix[missing_pct < 0.2, ]
cat("After filtering missing:", dim(meth_filtered), "\n")

# Step 7: Save
dir.create("data/raw", recursive=TRUE,
           showWarnings=FALSE)
write.csv(as.data.frame(meth_filtered),
          "data/raw/methylation_raw.csv")

cat("=====================================\n")
cat("METHYLATION SUCCESSFULLY SAVED\n")
cat("=====================================\n")
