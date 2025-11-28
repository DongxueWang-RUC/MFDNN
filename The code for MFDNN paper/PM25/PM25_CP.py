import os
import numpy as np
import pandas as pd
from tensorly.regression.cp_regression import CPRegressor
from sklearn.model_selection import train_test_split

np.random.seed(756)
data_path = "PM25"
frun = 50  

# covariate_data: 365 * T1 * T2 * 6
covariate_data = np.load(os.path.join(data_path, "interpolated_results_2022.npy"))
covariate_data = np.transpose(covariate_data, (0, 3, 1, 2))  # -> (365, 6, T1, T2)
y_data = np.load(os.path.join(data_path, "pm25_daily_means_2022.npy"))

train_indices = pd.read_csv(os.path.join(data_path, "train_indices_list.csv"), header=None).to_numpy()
test_indices  = pd.read_csv(os.path.join(data_path, "test_indices_list.csv"), header=None).to_numpy()

def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def CP_regression(X_train, y_train, X_test, rank, regularization=0.01):
    model = CPRegressor(weight_rank=rank, reg_W=regularization,
                        tol=1e-6, n_iter_max=200, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

def select_tensor_hyperparameters(X_train, y_train):
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=123
    )

    rank_values = [2]
    reg_values = [0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05, 0.07, 0.1]

    best_model = None
    best_rmse = np.inf
    best_hyper = None

    for rank in rank_values:
        for reg in reg_values:
            try:         
                model = CPRegressor(
                    weight_rank=rank,
                    reg_W=reg,
                    tol=1e-6,
                    n_iter_max=200,
                    verbose=0
                )
                model.fit(X_tr, y_tr)

                y_pred_val = model.predict(X_val)
                rmse_val = calculate_rmse(y_val, y_pred_val)

                if rmse_val < best_rmse:
                    best_rmse = rmse_val
                    best_hyper = {'rank': rank, 'regularization': reg}
                    best_model = model

            except:
                continue

    
    if best_model is None:
        rank = 2
        reg = 0.1
        best_model = CPRegressor(
            weight_rank=rank, reg_W=reg,
            tol=1e-6, n_iter_max=200, verbose=0
        )
        best_model.fit(X_train, y_train)
        best_hyper = {'rank': rank, 'regularization': reg}

    return best_hyper, best_model

def evaluate_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return calculate_rmse(y_test, y_pred)

all_rmse = []

for run_idx in range(frun):
    X_train = covariate_data[train_indices[run_idx], :, :, :]
    X_test  = covariate_data[test_indices[run_idx], :, :, :]
    y_train = y_data[train_indices[run_idx]]
    y_test  = y_data[test_indices[run_idx]]

    best_hyper, best_model = select_tensor_hyperparameters(X_train, y_train)
    
    rmse_test = evaluate_on_test(best_model, X_test, y_test)
    print(f"Run {run_idx+1}: Best hyperparameters -> rank={best_hyper['rank']}, "
        f"regularization={best_hyper['regularization']}, Test RMSE={rmse_test:.4f}")
    all_rmse.append(rmse_test)

rmse_mean = np.mean(all_rmse)
rmse_std  = np.std(all_rmse)

df_results = pd.DataFrame([{
    "rmse_mean": rmse_mean,
    "rmse_std": rmse_std
}])
save_path = os.path.join(data_path, "PM25_CP_results_50runs.csv")
df_results.to_csv(save_path, index=False)
