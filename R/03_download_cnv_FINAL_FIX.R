# R/03_download_cnv_FINAL_FIX.R

library(TCGAbiolinks)
library(SummarizedExperiment)

cat("=====================================\n")
cat("TCGA-BRCA CNV FINAL FIX\n")
cat("Remove normal samples + duplicates\n")
cat("=====================================\n")

# Step 1: Query
query_cnv <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Copy Number Variation",
  data.type = "Gene Level Copy Number"
)

# Step 2: Download (skip if already done)
GDCdownload(query_cnv, method="api", files.per.chunk=20)

# Step 3: Extract metadata
results <- query_cnv$results[[1]]

cat("Total files:", nrow(results), "\n")

# 🔥 Extract sample barcode
barcodes <- results$cases

# 🔥 Extract sample type (positions 14-15)
sample_type <- substr(barcodes, 14, 15)

# Keep ONLY tumor samples (01)
results_tumor <- results[sample_type == "01", ]

cat("Tumor samples:", nrow(results_tumor), "\n")

# 🔥 Extract patient ID (first 12 chars)
patient_id <- substr(results_tumor$cases, 1, 12)

# Remove duplicate patients
keep_idx <- !duplicated(patient_id)
results_clean <- results_tumor[keep_idx, ]

cat("After patient deduplication:", nrow(results_clean), "\n")

# Replace query results
query_cnv$results[[1]] <- results_clean

# Step 4: Prepare data
cat("Preparing CNV matrix...\n")
data_cnv <- GDCprepare(query_cnv)

# Step 5: Extract matrix
cnv_matrix <- assay(data_cnv)

cat("CNV matrix dimensions:", dim(cnv_matrix), "\n")

# Step 6: Save
dir.create("data/raw", recursive=TRUE, showWarnings=FALSE)

write.csv(as.data.frame(cnv_matrix),
          "data/raw/cnv_raw.csv")

cat("=====================================\n")
cat("CNV SUCCESSFULLY GENERATED\n")
cat("=====================================\n")
