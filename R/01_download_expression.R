# R/01_download_expression.R
library(TCGAbiolinks)
library(SummarizedExperiment)

cat("Downloading BRCA gene expression...\n")
cat("This takes 10-20 minutes...\n")

query_exp <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts"
)

GDCdownload(query_exp, method="api",
            files.per.chunk=10)

data_exp <- GDCprepare(query_exp)

expr_matrix <- assay(data_exp, "fpkm_unstrand")
cat("Expression matrix:", dim(expr_matrix), "\n")

clinical <- as.data.frame(colData(data_exp))
cat("Clinical samples:", nrow(clinical), "\n")
cat("Subtype column available:",
    "paper_BRCA_Subtype_PAM50" %in%
    colnames(clinical), "\n")

dir.create("data/raw", recursive=TRUE,
           showWarnings=FALSE)

write.csv(as.data.frame(expr_matrix),
          "data/raw/expression_raw.csv")

write.csv(clinical[, c("barcode",
    "paper_BRCA_Subtype_PAM50",
    "vital_status",
    "days_to_death",
    "days_to_last_follow_up")],
    "data/raw/clinical_subtypes.csv")

cat("Saved: data/raw/expression_raw.csv\n")
cat("Saved: data/raw/clinical_subtypes.csv\n")
cat("Expression download complete!\n")
