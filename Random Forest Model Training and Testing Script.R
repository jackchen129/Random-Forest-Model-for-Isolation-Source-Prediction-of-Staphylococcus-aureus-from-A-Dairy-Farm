# ================================================================
# 01_main_workflow.R
# Binary Source Prediction (Udder vs Other) using a Random Forest Model
# ================================================================

if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse, caret, ranger, readxl, here, janitor,
  ggplot2, RColorBrewer, pROC
)

set.seed(123)
output_dir <- here("Binary_Source_Results")
dir.create(output_dir, showWarnings = FALSE)

# --- Source external scripts -----------------------------------
source(here("scripts/02_data_preprocessing.R"))
source(here("scripts/03_model_training.R"))
source(here("scripts/04_model_evaluation.R"))

# --- Main workflow ----------------------------------------------
message("Running Binary Source Prediction (Udder vs Other)...")

data <- load_data()
stopifnot(!is.null(data))

processed_data <- preprocess_data(data)
stopifnot(!is.null(processed_data))

train_idx <- createDataPartition(processed_data$source, p = 0.75, list = FALSE)
train_data <- processed_data[train_idx, ]
test_data <- processed_data[-train_idx, ]

model <- train_model(train_data)
stopifnot(!is.null(model))

results <- evaluate_model(model, test_data)
stopifnot(!is.null(results))

# --- Save outputs ----------------------------------------------
write_csv(results$predictions, file.path(output_dir, "predictions.csv"))
write_csv(results$variable_importance, file.path(output_dir, "feature_importance.csv"))
saveRDS(model, file.path(output_dir, "binary_source_model.rds"))

sink(file.path(output_dir, "performance_summary.txt"))
cat("Binary Source Prediction: Udder vs Other\n")
cat("=============================================\n\n")
cat("Accuracy:", round(results$confusion_matrix$overall["Accuracy"], 3), "\n")
cat("Kappa:", round(results$confusion_matrix$overall["Kappa"], 3), "\n")
cat("AUC:", round(results$auc, 3), "\n\n")
cat("Confusion Matrix:\n")
print(results$confusion_matrix$table)
cat("\n\nTop Predictive Accessory Genes:\n")
print(head(results$variable_importance, 15))
sink()

message("✔️ Binary classification complete! Results saved to: ", normalizePath(output_dir))

# ================================================================
# 02_data_preprocessing.R
# Data loading and preprocessing functions
# ================================================================

load_data <- function() {
  tryCatch({
    stopifnot(file.exists("S. aureus.xlsx"))
    stopifnot(file.exists("Gene Presence Absence.csv"))
    
    message("Loading source metadata...")
    meta <- read_excel("S. aureus.xlsx") %>%
      clean_names() %>%
      rename(Isolate = isolate) %>%
      mutate(source = ifelse(source == "Udder", "Udder", "Other")) %>%
      mutate(source = factor(source, levels = c("Other", "Udder")))  # reference = "Other"
    
    message("Loading gene presence/absence matrix...")
    acc_raw <- read_csv("Gene Presence Absence.csv", col_types = cols(.default = "c"))
    gene_names <- acc_raw$Gene
    
    acc_matrix <- acc_raw %>%
      select(-Gene) %>%
      mutate(across(everything(), ~ ifelse(. == "1", 1, 0))) %>%
      as.matrix()
    rownames(acc_matrix) <- gene_names
    
    acc_matrix_t <- as.data.frame(t(acc_matrix))
    colnames(acc_matrix_t) <- make_clean_names(gene_names)
    acc_matrix_t <- acc_matrix_t %>%
      rownames_to_column("Isolate")
    
    message("Merging metadata and gene matrix...")
    combined <- inner_join(meta, acc_matrix_t, by = "Isolate")
    
    if (nrow(combined) == 0) stop("No matched isolates after merge")
    message(paste("Final sample count:", nrow(combined)))
    print(combined %>% count(source))
    
    return(combined)
  }, error = function(e) {
    message("Error during data load: ", e$message)
    return(NULL)
  })
}

preprocess_data <- function(data) {
  tryCatch({
    gene_cols <- setdiff(names(data), c("Isolate", "source"))
    nzv <- nearZeroVar(select(data, all_of(gene_cols)), names = TRUE)
    
    excluded_genes <- nzv
    included_genes <- setdiff(gene_cols, excluded_genes)
    
    if (length(excluded_genes) > 0) {
      message("Removing ", length(excluded_genes), " near-zero variance features")
      data <- select(data, -all_of(excluded_genes))
    }
    
    data <- data[, colSums(is.na(data)) < nrow(data)]
    
    write_lines(included_genes, file.path(output_dir, "included_genes.txt"))
    write_lines(excluded_genes, file.path(output_dir, "excluded_genes.txt"))
    
    message("Included genes: ", length(included_genes))
    message("Excluded genes: ", length(excluded_genes))
    
    return(data)
  }, error = function(e) {
    message("Error in preprocessing: ", e$message)
    return(NULL)
  })
}

# ================================================================
# 03_model_training.R
# Random forest training function
# ================================================================

train_model <- function(train_data) {
  tryCatch({
    ctrl <- trainControl(
      method = "cv", number = 5,
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      allowParallel = TRUE,
      savePredictions = "final"
    )
    
    model <- train(
      x = select(train_data, -Isolate, -source),
      y = train_data$source,
      method = "ranger",
      importance = "permutation",
      num.trees = 500,
      metric = "ROC",
      trControl = ctrl
    )
    
    return(model)
  }, error = function(e) {
    message("Error in model training: ", e$message)
    return(NULL)
  })
}

# ================================================================
# 04_model_evaluation.R
# Model evaluation and visualization functions
# ================================================================

evaluate_model <- function(model, test_data) {
  tryCatch({
    preds_prob <- predict(model, test_data, type = "prob")
    preds_class <- predict(model, test_data)
    
    preds <- bind_cols(
      SampleID = test_data$Isolate,
      actual = test_data$source,
      preds_prob,
      predicted = preds_class
    )
    
    cm <- confusionMatrix(preds$predicted, preds$actual, positive = "Udder")
    
    var_imp <- varImp(model)$importance %>%
      rownames_to_column("Feature") %>%
      arrange(desc(Overall))
    
    roc_obj <- roc(response = preds$actual,
                   predictor = preds$Udder,
                   levels = c("Other", "Udder"))
    
    plots <- list()
    plots$roc <- ggroc(roc_obj) +
      ggtitle("ROC Curve: Udder vs Other") +
      theme_minimal() +
      geom_abline(linetype = "dashed", color = "gray")
    
    plots$imp <- var_imp %>%
      head(15) %>%
      ggplot(aes(reorder(Feature, Overall), Overall)) +
      geom_col(fill = "#377EB8") +
      coord_flip() +
      labs(title = "Top 15 Predictive Genes", x = "", y = "Importance") +
      theme_minimal()
    
    ggsave(file.path(output_dir, "roc_curve.png"), plots$roc, width = 6, height = 5)
    ggsave(file.path(output_dir, "feature_importance.png"), plots$imp, width = 8, height = 6)
    
    return(list(
      predictions = preds,
      confusion_matrix = cm,
      variable_importance = var_imp,
      auc = auc(roc_obj)
    ))
  }, error = function(e) {
    message("Error in evaluation: ", e$message)
    return(NULL)
  })
}
