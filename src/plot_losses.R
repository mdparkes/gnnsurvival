library(ggplot2)


project_dir <- getwd()
expt_name <- commandArgs(trailingOnly = T)
if (expt_name == "experiment1") {
  cancer_types <- c("BLCA", "COAD", "GBM", "HNSC", "KIRC", "LGG", 
                    "LIHC", "LUAD", "LUSC", "OV", "SKCM", "STAD")
} else {
  cancer_types <- c("KIRC", "LGG", "LUAD", "SKCM")
}
max_epochs <- 150
for (cancer in cancer_types) {
  dir <- paste(project_dir, expt_name, cancer, "exports", sep = "/")
  fname <- paste0(dir, "/", cancer, "_losses_2.csv")
  
  # Load and format data ####
  losses <- read.csv(fname)
  losses <- losses[ ,-1]
  losses$epoch <- losses$epoch + 1
  losses$model_type <- factor(
    losses$model_type, 
    levels = c("GNN", "MLP")
  )
  losses$train_or_test <- factor(
    losses$train_or_test, 
    levels = c("Train", "Test")
  )
  losses$feature_selection <- factor(
    losses$feature_selection,
    levels = c("No feature selection", "Feature selection")
  )
  losses$fold <- factor(
    sapply(losses$fold, function(x) paste("Fold", x)),
    levels = paste("Fold", seq_len(max(losses$fold)))
  )
  losses$model <- factor(
    paste0(losses$model_type, " - ", losses$feature_selection),
    levels = c(
      "GNN - No feature selection",
      "GNN - Feature selection",
      "MLP - No feature selection",
      "MLP - Feature selection"
    )
  )
  
  # Generate Plot ####
  ggplot(losses, aes(x = epoch, y = loss, color = train_or_test)) +
    geom_line() +
    facet_grid(model ~ fold, scales = "free_y") +
    labs(x = "Epoch", y = "Loss (negative log likelihood)", color = "Data") +
    theme(panel.border = element_rect(color = "black", fill = NA,
                                      linewidth = 0.5), 
          strip.background = element_rect(color = "black", linewidth = 0.5),
          strip.text = element_text(size = 7))
  ggsave(
    filename = paste0(project_dir, "/", expt_name, "/", cancer, "/",
                      cancer, "_losses.png"),
    plot = last_plot(),
    width = 8, height = 6, units = "in"
  )
}




