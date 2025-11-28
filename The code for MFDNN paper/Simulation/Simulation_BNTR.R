library(BNTR)
library(reticulate)
library(parallel)
library(doSNOW)
library(pbapply)
set.seed(123)

quiet <- function(expr) {
  sink(tempfile())
  on.exit(sink())
  force(expr)
}


split_data <- function(X_array, y_array) {
  p <- dim(X_array)[1]
  N <- dim(X_array)[2]
  T1 <- dim(X_array)[3]
  T2 <- dim(X_array)[4]
  
  # p * T1 * T2 * N
  X_reordered <- aperm(X_array, c(1, 3, 4, 2))
  
  train_size <- floor(N * 0.4)
  val_size   <- floor(N * 0.1)
  
  idx <- sample(N)
  
  list(
    X_train = X_reordered[, , , idx[1:train_size], drop = FALSE],
    y_train = y_array[idx[1:train_size]],
    X_val   = X_reordered[, , , idx[(train_size + 1):(train_size + val_size)], drop = FALSE],
    y_val   = y_array[idx[(train_size + 1):(train_size + val_size)]],
    X_test  = X_reordered[, , , idx[(train_size + val_size + 1):N], drop = FALSE],
    y_test  = y_array[idx[(train_size + val_size + 1):N]]
  )
}

select_bntr <- function(X_train, y_train, X_val, y_val, X_test, y_test,
                        R, alpha, lambda) {
  
  out <- quiet(validation_broadcasted_sparsetenreg(
    R, alpha, lambda,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    num_knots = 5,
    order = 4
  ))
  

  table = out$b_validation_test_lambda_R_nonlinear
  best_idx = which.min(table[, 2])
  
  test_mse = table[best_idx, 3]
  test_rmse = sqrt(test_mse)
  nrmse = test_rmse / sd(y_test)
  
  return(nrmse)
}


run_single_config <- function(config_run) {

  quiet_local <- function(expr) {
    sink(tempfile())
    on.exit(sink())
    force(expr)
  }
  
  cfg <- config_run$cfg
  yid <- config_run$yid
  r <- config_run$r
  Xlist <- config_run$Xlist
  ylist <- config_run$ylist
  R_values <- config_run$R_values
  alpha_values <- config_run$alpha_values
  lambda_values <- config_run$lambda_values
  
  Tval <- cfg$T
  nval <- cfg$n
  
  Xa <- Xlist[r, , , , ]
  ya <- ylist[r, yid, ]

  split_data_local <- function(X_array, y_array) {
    p <- dim(X_array)[1]
    N <- dim(X_array)[2]
    T1 <- dim(X_array)[3]
    T2 <- dim(X_array)[4]
    
    # p * T1 * T2 * N
    X_reordered <- aperm(X_array, c(1, 3, 4, 2))
    
    train_size <- floor(N * 0.4)
    val_size   <- floor(N * 0.1)
    
    idx <- sample(N)
    
    list(
      X_train = X_reordered[, , , idx[1:train_size], drop = FALSE],
      y_train = y_array[idx[1:train_size]],
      X_val   = X_reordered[, , , idx[(train_size + 1):(train_size + val_size)], drop = FALSE],
      y_val   = y_array[idx[(train_size + 1):(train_size + val_size)]],
      X_test  = X_reordered[, , , idx[(train_size + val_size + 1):N], drop = FALSE],
      y_test  = y_array[idx[(train_size + val_size + 1):N]]
    )
  }
  
  sp <- split_data_local(Xa, ya)
  

  out <- quiet_local(BNTR::validation_broadcasted_sparsetenreg(
    R_values, alpha_values, lambda_values,
    sp$X_train, sp$y_train,
    sp$X_val, sp$y_val,
    sp$X_test, sp$y_test,
    num_knots = 5,
    order = 4
  ))
  

  table = out$b_validation_test_lambda_R_nonlinear
  best_idx = which.min(table[, 2]) 
  
  test_mse = table[best_idx, 3]
  test_rmse = sqrt(test_mse)
  nrmse = test_rmse / sd(sp$y_test)
  
  return(list(
    config = paste0("T", Tval, "_n", nval),
    yid = yid,
    run = r,
    nrmse = nrmse
  ))
}


run_bntr_experiment_parallel <- function(frun = 5, cores = NULL, T_values = c(16,32)) {
  
  if (is.null(cores)) cores <- detectCores() - 1
  
  # 配置列表
  configs <- list(
    list(T = 16, n = 200),
    list(T = 16, n = 400),
    list(T = 32, n = 200),
    list(T = 32, n = 400)
  )
  

  configs <- configs[sapply(configs, function(cfg) cfg$T %in% T_values)]
  
  np <- import("numpy")
  path <- "Simulation"
  
  all_tasks <- list()
  
  for (cfg in configs) {
    Tval <- cfg$T
    nval <- cfg$n
    key <- paste0("T", Tval, "_n", nval)
    
    cat("Loading data for:", key, "\n")
    
    Xlist <- np$load(file.path(path, paste0("Xlist_T", Tval, "_n", nval, ".npy")), allow_pickle = TRUE)
    ylist <- np$load(file.path(path, paste0("ylist_T", Tval, "_n", nval, ".npy")), allow_pickle = TRUE)
    

    if (Tval == 16) {
      # For small sample sizes (T=16), empirical findings show that tuning R and alpha has limited impact on results
      # Therefore, fix R=2 and alpha=0.5, focusing primarily on lambda parameter optimization
      R_values <- 2; alpha_values <- 0.5; lambda_values <- c(0.01,0.03,0.05,0.07,0.1)
    } else if (Tval == 32) {
      # For T=32, experimental results show that larger lambda values cause algorithm failure
      # Therefore, use a small fixed lambda=0.01 while tuning R and alpha parameters
      R_values <- c(2,3); alpha_values <- c(0,0.5,1); lambda_values <- 0.01 
    }
    
    for (yid in 1:6) {
      for (r in 1:frun) {
        all_tasks[[length(all_tasks) + 1]] <- list(
          cfg = cfg, yid = yid, r = r,
          Xlist = Xlist, ylist = ylist,
          R_values = R_values,
          alpha_values = alpha_values,
          lambda_values = lambda_values
        )
      }
    }
  }
  
  cat("Total tasks:", length(all_tasks), "\n")
  cat("Using", cores, "cores for parallel computation\n")
  
  cl <- makeCluster(cores)
  clusterEvalQ(cl, { library(BNTR); library(reticulate) })
  
  results_parallel <- pblapply(all_tasks, run_single_config, cl = cl)
  
  stopCluster(cl)

  results <- list()
  for (res in results_parallel) {
    config_key <- res$config
    yid_key <- paste0("y", res$yid)
    if (!config_key %in% names(results)) results[[config_key]] <- list()
    if (!yid_key %in% names(results[[config_key]])) results[[config_key]][[yid_key]] <- list(all_nrmse = numeric(frun))
    results[[config_key]][[yid_key]]$all_nrmse[res$run] <- res$nrmse
  }

  for (config_key in names(results)) {
    for (yid_key in names(results[[config_key]])) {
      nrmse_vec <- results[[config_key]][[yid_key]]$all_nrmse
      results[[config_key]][[yid_key]]$nrmse_mean <- mean(nrmse_vec)
      results[[config_key]][[yid_key]]$nrmse_std <- sd(nrmse_vec)
    }
  }
  
  return(results)
}
##### T=16 #####
bntr_results_T16 <- run_bntr_experiment_parallel(frun = 50, cores = 10, T_values = 16)


results_list <- list()
for (config_key in names(bntr_results_T16)) {
  for (yid_key in names(bntr_results_T16[[config_key]])) {
    res <- bntr_results_T16[[config_key]][[yid_key]]
    results_list[[length(results_list)+1]] <- data.frame(
      config = config_key,
      yid = yid_key,
      nrmse_mean = res$nrmse_mean,
      nrmse_sd = res$nrmse_std
    )
  }
}
results_df <- do.call(rbind, results_list)

current_dir <- getwd()
output_dir <- file.path(current_dir, "Simulation")
dir.create(output_dir, showWarnings = FALSE)

write.csv(results_df, file = file.path(output_dir, "BNTR_T16_results.csv"), row.names = FALSE)
saveRDS(bntr_results_T16, file = file.path(output_dir, "BNTR_T16_detailed.rds"))

##### T=32 #####
bntr_results_T32 <- run_bntr_experiment_parallel(frun = 50, cores = 10, T_values = 32)


results_list <- list()
for (config_key in names(bntr_results_T32)) {
  for (yid_key in names(bntr_results_T32[[config_key]])) {
    res <- bntr_results_T32[[config_key]][[yid_key]]
    results_list[[length(results_list)+1]] <- data.frame(
      config = config_key,
      yid = yid_key,
      nrmse_mean = res$nrmse_mean,
      nrmse_sd = res$nrmse_std
    )
  }
}
results_df <- do.call(rbind, results_list)

current_dir <- getwd()
output_dir <- file.path(current_dir, "Simulation")
dir.create(output_dir, showWarnings = FALSE)

write.csv(results_df, file = file.path(output_dir, "BNTR_T32_results.csv"), row.names = FALSE)
saveRDS(bntr_results_T16, file = file.path(output_dir, "BNTR_T32_detailed.rds"))