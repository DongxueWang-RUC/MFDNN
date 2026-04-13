library(BNTR)
library(reticulate)

set.seed(756)
setwd("/Users/wangdongxue/Desktop/MFDNN_STCO_revise/MFDNN_code/EEG")
quiet <- function(expr) {
  sink(tempfile())
  on.exit(sink())
  force(expr)
}

# =========================================================
# Paths
# If run as script, use script path;
# otherwise use current working directory.
# =========================================================
get_base_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  script_path <- sub(file_arg, "", args[grep(file_arg, args)])
  
  if (length(script_path) > 0) {
    return(dirname(normalizePath(script_path)))
  } else {
    return(normalizePath(getwd()))
  }
}

eeg_dir <- get_base_dir()
data_dir <- file.path(eeg_dir, "Data")
results_dir <- file.path(eeg_dir, "Results")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# =========================================================
# Data
# =========================================================
train_indices_list <- read.csv(
  file.path(data_dir, "train_indices_list.csv"),
  header = TRUE
)
test_indices_list <- read.csv(
  file.path(data_dir, "test_indices_list.csv"),
  header = TRUE
)

EEG_x <- readRDS(file.path(data_dir, "EEG_y.rds"))   # 64 * 64 * n
EEG_y <- c(rep(1, 39), rep(-1, 22))                  # 61 subjects

RUNS <- 50

R_values <- 2
alpha_values <- c(0, 0.5, 1)
lambda_values <- c(0.01, 0.05, 0.1, 0.5, 1)

all_f1 <- numeric(RUNS)

for (run_idx in 1:RUNS) {
  print(run_idx)
  
  train_id <- train_indices_list[, run_idx]
  test_id  <- test_indices_list[, run_idx]
  
  train_id <- train_id[!is.na(train_id)]
  test_id  <- test_id[!is.na(test_id)]
  
  X_train <- EEG_x[, , train_id, drop = FALSE]
  y_train <- EEG_y[train_id]
  
  X_test  <- EEG_x[, , test_id, drop = FALSE]
  y_test  <- EEG_y[test_id]
  
  n_train <- length(train_id)
  n_val <- max(1, floor(n_train * 0.25))
  
  val_id <- sample(train_id, n_val)
  pure_train_id <- setdiff(train_id, val_id)
  
  X_train_pure <- EEG_x[, , pure_train_id, drop = FALSE]
  y_train_pure <- EEG_y[pure_train_id]
  
  X_val <- EEG_x[, , val_id, drop = FALSE]
  y_val <- EEG_y[val_id]
  
  out <- quiet(BNTR::validation_broadcasted_sparsetenreg(
    R_values, alpha_values, lambda_values,
    X_train_pure, y_train_pure,
    X_val, y_val,
    X_test, y_test,
    num_knots = 5,
    order = 4
  ))
  
  # F1
  n_test <- length(y_test)
  num_knots <- 5
  knots <- quantile(c(X_train_pure), probs = seq(0, 1, length.out = num_knots))
  
  Phi_test <- tildePhiX_trans(X_test, knots)
  Phi_test <- matrix(Phi_test, nrow = prod(dim(out$BB)), ncol = n_test)
  
  y_pred <- out$bb + crossprod(Phi_test, as.vector(out$BB))
  y_pred <- ifelse(y_pred > 0, 1, -1)
  
  tp <- sum(y_pred == 1  & y_test ==  1)
  fp <- sum(y_pred == 1  & y_test == -1)
  fn <- sum(y_pred == -1 & y_test ==  1)
  
  precision <- tp / (tp + fp + 1e-10)
  recall <- tp / (tp + fn + 1e-10)
  f1 <- 2 * precision * recall / (precision + recall + 1e-10)
  
  all_f1[run_idx] <- f1
}

mean_f1 <- mean(all_f1)
sd_f1 <- sd(all_f1)

out_path <- file.path(results_dir, "EEG_BNTR.csv")

write.csv(
  data.frame(mean_f1 = mean_f1, sd_f1 = sd_f1),
  file = out_path,
  row.names = FALSE
)

cat("\nSaved to:", out_path, "\n")