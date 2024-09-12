ttfn <- function(x, y, ...) {
  tryCatch(
    t.test(x, y, ...)$p.value,
    error = function(e) NA
  )
}

wtfn <- function(x, y, ...) {
  tryCatch(
    wilcox.test(x, y, ...)$p.value,
    error = function(e) NA
  )
}

args <- commandArgs(trailingOnly = T)
expt_dir <- args[1]
if (!grepl("/$", expt_dir)) {
  expt_dir <- paste0(expt_dir, "/")
}

if (grepl("experiment1", expt_dir)) {
  # Results without feature selection
  gnn_conc_pth <- paste0(expt_dir,
                         "gnn_reactome_unmerged_directed_nonrelational_",
                         "feature-selection=no-sagpool_binary-auc.csv")
  mlp_conc_pth <- paste0(expt_dir,
                         "mlp_reactome_directed_",
                         "feature-selection=no-sagpool_binary-auc.csv")
  gnn_conc <- read.csv(gnn_conc_pth)[,-1]  # Folds x Cancer types
  mlp_conc <- read.csv(mlp_conc_pth)[,-1]  # Folds x Cancer types
  
  # Results with feature selection
  gnn_conc_fs_pth <- paste0(expt_dir,
                            "gnn_reactome_unmerged_directed_nonrelational_",
                            "feature-selection=sagpool_binary-auc.csv")
  mlp_conc_fs_pth <- paste0(expt_dir,
                            "mlp_reactome_directed_",
                            "feature-selection=sagpool_binary-auc.csv")
  gnn_conc_fs <- read.csv(gnn_conc_fs_pth)[,-1]  # Folds x Cancer types
  mlp_conc_fs <- read.csv(mlp_conc_fs_pth)[,-1]  # Folds x Cancer types
  
  
  # Compare MLP and GNN models both without feature selection
  # Perform unpaired, one-sided Welch's t-tests favoring MLP or GNN
  res1 <- mapply(ttfn, gnn_conc, mlp_conc, alternative = "two.sided")
  res2 <- mapply(ttfn, gnn_conc_fs, mlp_conc_fs, alternative = "two.sided")
  res <- as.data.frame(rbind(res1, res2))
  row.names(res) <- c("no-sagpool", "sagpool")
  file_out <- paste0(
    expt_dir, "binary-auc_unpaired_two-sided_welch_pvals.csv"
  )
  write.csv(res, file_out)
  
  res1 <- mapply(ttfn, gnn_conc, mlp_conc, alternative = "greater")
  res2 <- mapply(ttfn, gnn_conc_fs, mlp_conc_fs, alternative = "greater")
  res <- as.data.frame(rbind(res1, res2))
  row.names(res) <- c("no-sagpool", "sagpool")
  file_out <- paste0(
    expt_dir, "binary-auc_unpaired_gnn-greater_welch_pvals.csv"
  )
  write.csv(res, file_out)
  
  res1 <- mapply(ttfn, gnn_conc, mlp_conc, alternative = "less")
  res2 <- mapply(ttfn, gnn_conc_fs, mlp_conc_fs, alternative = "less")
  res <- as.data.frame(rbind(res1, res2))
  row.names(res) <- c("no-sagpool", "sagpool")
  file_out <- paste0(
    expt_dir, "binary-auc_unpaired_gnn-less_welch_pvals.csv"
  )
  write.csv(res, file_out)
  
  # Test whether the AUC scores of models trained in different cancer
  # types differ depending on whether feature selection was used.
  gnn_conc_t <- as.data.frame(t(gnn_conc))
  gnn_conc_fs_t <- as.data.frame(t(gnn_conc_fs))
  mlp_conc_t <- as.data.frame(t(mlp_conc))
  mlp_conc_fs_t <- as.data.frame(t(mlp_conc_fs))
  gnn_fs_res <- mapply(wtfn, gnn_conc_t, gnn_conc_fs_t,
                       alternative = "two.sided", exact = F, paired = T)
  gnn_fs_res <- p.adjust(gnn_fs_res, "fdr")
  gnn_fs_res <- setNames(gnn_fs_res, paste0("fold", seq_along(names(gnn_fs_res))))
  mlp_fs_res <- mapply(wtfn, mlp_conc_t, mlp_conc_fs_t,
                       alternative = "two.sided", exact = F, paired = T)
  mlp_fs_res <- p.adjust(mlp_fs_res, "fdr")
  mlp_fs_res <- setNames(mlp_fs_res, paste0("fold", seq_along(names(mlp_fs_res))))
  fs_res <- as.data.frame(rbind(gnn_fs_res, mlp_fs_res))
  row.names(fs_res) <- c("GNN", "MLP")
  file_out <- paste0(
    expt_dir, "binary-auc_gnn_mlp_feature-selection_paired_wilcoxon_fdr.csv"
  )
  write.csv(fs_res, file_out)
} else {
  # Results without feature selection (MLP only)
  mlp_conc_pth <- paste0(expt_dir,
                         "mlp_reactome_directed_",
                         "feature-selection=no-sagpool_binary-auc.csv")
  mlp_conc <- read.csv(mlp_conc_pth)[,-1]  # Folds x Cancer types
  
  # Results with feature selection
  gnn_conc_fs_pth <- paste0(expt_dir,
                            "gnn_reactome_unmerged_directed_nonrelational_",
                            "feature-selection=sagpool_binary-auc.csv")
  mlp_conc_fs_pth <- paste0(expt_dir,
                            "mlp_reactome_directed_",
                            "feature-selection=sagpool_binary-auc.csv")
  gnn_conc_fs <- read.csv(gnn_conc_fs_pth)[,-1]  # Folds x Cancer types
  mlp_conc_fs <- read.csv(mlp_conc_fs_pth)[,-1]  # Folds x Cancer types
  
  res1 <- mapply(ttfn, gnn_conc_fs, mlp_conc, alternative = "two.sided")
  res2 <- mapply(ttfn, gnn_conc_fs, mlp_conc_fs, alternative = "two.sided")
  res <- as.data.frame(cbind(res1, res2))
  colnames(res) <- c("GNN-FS vs MLP-NOFS", "GNN-FS vs MLP-FS")
  file_out <- paste0(
    expt_dir, "binary-auc_unpaired_two-sided_welch_pvals.csv"
  )
  write.csv(res, file_out)
  
  # Perorm unpaired, one-sided Welch's t-tests with null hypothesis that the
  # mean GNN AUC for a cancer type is less than the MLP's
  res1 <- mapply(ttfn, gnn_conc_fs, mlp_conc, alternative = "greater")
  res2 <- mapply(ttfn, gnn_conc_fs, mlp_conc_fs, alternative = "greater")
  res <- as.data.frame(cbind(res1, res2))
  colnames(res) <- c("GNN-FS vs MLP-NOFS", "GNN-FS vs MLP-FS")
  file_out <- paste0(
    expt_dir, "binary-auc_unpaired_gnn-greater_welch_pvals.csv"
  )
  write.csv(res, file_out)
  
  # Perform unpaired, one-sided Welch's t-tests with null hypothesis that the
  # mean GNN AUC for a cancer type is greater than the MLP's
  res1 <- mapply(ttfn, gnn_conc_fs, mlp_conc, alternative = "less")
  res2 <- mapply(ttfn, gnn_conc_fs, mlp_conc_fs, alternative = "less")
  res <- as.data.frame(cbind(res1, res2))
  colnames(res) <- c("MLP-NOFS vs GNN-FS", "MLP-FS vs GNN-FS")
  file_out <- paste0(
    expt_dir, "binary-auc_unpaired_gnn-less_welch_pvals.csv"
  )
  write.csv(res, file_out)
}
