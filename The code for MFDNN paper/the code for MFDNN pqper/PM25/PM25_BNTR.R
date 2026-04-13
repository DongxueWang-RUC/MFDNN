library(BNTR)
library(reticulate)
library(parallel)
library(doSNOW)
library(pbapply)
library(RcppCNPy)
library(readr)

set.seed(123)

quiet <- function(expr) {
  sink(tempfile())
  on.exit(sink())
  force(expr)
}
setwd('/Users/wangdongxue/Desktop/MFDNN_STCO_revise/MFDNN_code/PM25') 
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

pm25_dir <- get_base_dir()
data_dir <- file.path(pm25_dir, "Data")
results_dir <- file.path(pm25_dir, "Results")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# =========================================================
# Data
# =========================================================
np <- import("numpy")

covariate_data <- np$load(
  file.path(data_dir, "covariates_10x10.npy"),
  allow_pickle = TRUE
) # N * T1 * T2 * p

X <- aperm(covariate_data, c(4, 2, 3, 1)) # p * T1 * T2 * N
y <- npyLoad(file.path(data_dir, "pm25_daily_means_2022.npy")) # 365

train_mat <- as.matrix(
  read_csv(file.path(data_dir, "train_indices_list.csv"), col_names = FALSE)
)
test_mat <- as.matrix(
  read_csv(file.path(data_dir, "test_indices_list.csv"), col_names = FALSE)
)

train_mat <- train_mat + 1
test_mat <- test_mat + 1

# =========================================================
# BNTR runner
# =========================================================
run_single_bntr <- function(i, X, y, train_mat, test_mat,
                            R_values, alpha_values, lambda_values) {
  set.seed(i)
  
  train_id <- train_mat[i, ]
  test_id  <- test_mat[i, ]
  
  train_id <- train_id[!is.na(train_id)]
  test_id  <- test_id[!is.na(test_id)]
  
  n_train <- length(train_id)
  n_val <- max(1, floor(n_train * 0.25))
  
  val_id <- sample(train_id, n_val)
  pure_train_id <- setdiff(train_id, val_id)
  
  X_train <- X[, , , pure_train_id, drop = FALSE]
  y_train <- y[pure_train_id]
  
  X_val <- X[, , , val_id, drop = FALSE]
  y_val <- y[val_id]
  
  X_test <- X[, , , test_id, drop = FALSE]
  y_test <- y[test_id]
  
  out <- quiet(BNTR::validation_broadcasted_sparsetenreg(
    R_values, alpha_values, lambda_values,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    num_knots = 5,
    order = 4
  ))
  
  tab <- out$b_validation_test_lambda_R_nonlinear
  best_idx <- which.min(tab[, 2])
  
  test_mse <- tab[best_idx, 3]
  test_rmse <- sqrt(test_mse)
  
  return(test_rmse)
}

# =========================================================
# Hyperparameter settings
# =========================================================
R_values <- 2
alpha_values <- c(0, 0.5, 1)
lambda_values <- c(0.01, 0.05, 0.1, 0.5, 1)

RUNS <- 50

# =========================================================
# Main
# =========================================================
results <- pbsapply(
  1:RUNS,
  function(i) {
    run_single_bntr(i, X, y, train_mat, test_mat,
                    R_values, alpha_values, lambda_values)
  }
)

rmse_mean <- mean(results)
rmse_sd <- sd(results)

summary_df <- data.frame(
  rmse_mean = rmse_mean,
  rmse_std = rmse_sd
)

save_path <- file.path(results_dir, "PM25_BNTR.csv")
write.csv(summary_df, file = save_path, row.names = FALSE)

cat("\nSaved to:", save_path, "\n")