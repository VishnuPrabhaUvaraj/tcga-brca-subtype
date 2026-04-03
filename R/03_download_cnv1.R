# R/03_download_cnv.R
library(TCGAbiolinks)
library(SummarizedExperiment)

cat("Downloading BRCA copy number variation...\n")

query_cnv <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Copy Number Variation",
  data.type = "Gene Level Copy Number"
)

GDCdownload(query_cnv, method="api",
            files.per.chunk=20)

# Fix: summarize.files=TRUE handles duplicate samples
data_cnv <- GDCprepare(query_cnv, summarize.files = TRUE)
cnv_matrix <- assay(data_cnv)
cat("CNV matrix:", dim(cnv_matrix), "\n")

write.csv(as.data.frame(cnv_matrix),
          "data/raw/cnv_raw.csv")
cat("Saved: data/raw/cnv_raw.csv\n")
cat("CNV download complete!\n")
