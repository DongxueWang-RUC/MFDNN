library(BNTR)
library(reticulate)
library(parallel)
library(doSNOW)
library(pbapply)
library(RcppCNPy)   # for reading .npy
library(readr)      # for reading CSV
set.seed(123)

quiet <- function(expr) {
  sink(tempfile())
  on.exit(sink())
  force(expr)
}

data_path <- "/Users/wangdongxue/Documents/MFDNN/MFDNN/PM25"

np <- import("numpy")
covariate_data <- np$load(file.path(data_path, "interpolated_results_2022.npy"), allow_pickle = TRUE) # N*T1*T2*p
X <- aperm(covariate_data, c(4, 2, 3, 1)) # p * T1 * T2 * N
y <- npyLoad(file.path(data_path, "pm25_daily_means_2022.npy")) # 365

train_mat <- as.matrix(read_csv(file.path(data_path, "train_indices_list.csv"), col_names = FALSE))
test_mat  <- as.matrix(read_csv(file.path(data_path, "test_indices_list.csv"), col_names = FALSE))

# R 中索引变成从1开始
train_mat <- train_mat + 1
test_mat  <- test_mat + 1

# -------------------------
# 单次 BNTR 运行逻辑
# -------------------------
run_single_bntr <- function(i, X, y, train_mat, test_mat,
                            R_values, alpha_values, lambda_values) {
  
  set.seed(i)   # 每次实验有固定验证划分
  
  train_id <- train_mat[i, ]
  test_id  <- test_mat[i, ]
  
  train_id <- train_id[!is.na(train_id)]
  test_id  <- test_id[!is.na(test_id)]
  
  # ---------- 验证集 = 训练集 25% ----------
  n_train <- length(train_id)
  n_val   <- max(1, floor(n_train * 0.25))
  
  val_id <- sample(train_id, n_val)
  pure_train_id <- setdiff(train_id, val_id)
  
  # ---------- 切三份数据 ----------
  X_train <- X[, , , pure_train_id, drop = FALSE]
  y_train <- y[pure_train_id]
  
  X_val   <- X[, , , val_id, drop = FALSE]
  y_val   <- y[val_id]
  
  X_test  <- X[, , , test_id, drop = FALSE]
  y_test  <- y[test_id]
  
  # ---------- 训练 BNTR ----------
  out <- quiet(BNTR::validation_broadcasted_sparsetenreg(
    R_values, alpha_values, lambda_values,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    num_knots = 5,
    order = 4
  ))
  
  tab <- out$b_validation_test_lambda_R_nonlinear
  
  # ---------- 基于验证误差选参数 ----------
  best_idx <- which.min(tab[, 2])  # 第2列是验证误差
  
  test_mse  <- tab[best_idx, 3]
  test_rmse <- sqrt(test_mse)

  
  return(test_rmse)
}

# -------------------------
# 超参数
# -------------------------
R_values     <- 2
alpha_values <- c(0, 0.5, 1)
lambda_values <- c(0.01, 0.05, 0.1, 0.5, 1)

RUNS <- 50

# -------------------------
# 运行 50 次
# -------------------------
results <- pbsapply(
  1:RUNS,
  function(i) {
    run_single_bntr(i, X, y, train_mat, test_mat, R_values, alpha_values, lambda_values)
  }
)


rmse_mean <- mean(results)
rmse_sd   <- sd(results)

# -------------------------
# 写入 CSV（只保存均值和标准差）
# -------------------------
summary_df <- data.frame(
  mean_rmse = rmse_mean,
  sd_rmse   = rmse_sd
)

data_path <- "/Users/wangdongxue/Documents/MFDNN/MFDNN/PM25"
write.csv(
  summary_df,
  file = file.path(data_path, "PM25_BNTR.csv"),
  row.names = FALSE
)

cat("50 次 RMSE 均值: ", rmse_mean, "\n")
cat("50 次 RMSE 标准差:", rmse_sd, "\n")
cat("结果已保存到: ", file.path(data_path, "PM25_BNTR.csv"), "\n")

