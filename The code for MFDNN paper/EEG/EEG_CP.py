import os
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from tensorly.regression.cp_regression import CPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


np.random.seed(756)


data_path = "/Users/wangdongxue/Documents/MFDNN/MFDNN/EEG"
train_indices_list = pd.read_csv(os.path.join(data_path, "train_indices_list.csv"))
test_indices_list  = pd.read_csv(os.path.join(data_path, "test_indices_list.csv"))
rds_file = os.path.join(data_path, "EEG_y.rds")
r_array = robjects.r['readRDS'](rds_file)

# 转换为 NumPy 数组
EEG_x = np.array(r_array) # 64*64*n
EEG_y = np.array([1]*39 + [-1]*22)  # 示例标签


frun = 50  # 重复次数


def evaluate_f1_cp(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_labels = np.where(y_pred > 0, 1, -1)
    return f1_score(y_test, y_labels, pos_label=1)


def select_cp_hyperparameters(X_train, y_train, rank_values, reg_values, val_ratio=0.25):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=756)
    
    best_model = None
    best_rmse = np.inf
    best_hyper = None

    for rank in rank_values:
        for reg in reg_values:
            try:
                model = CPRegressor(weight_rank=rank, reg_W=reg, tol=1e-6, n_iter_max=200, verbose=0)
                model.fit(X_tr, y_tr)
                y_val_pred = model.predict(X_val)
                mse_val = np.mean((y_val - y_val_pred)**2)

                if mse_val < best_rmse:
                    best_rmse = mse_val
                    best_hyper = {'rank': rank, 'regularization': reg}
                    best_model = model
            except:
                continue

    # 在整个训练集上重新训练
    if best_model is not None:
        best_model.fit(X_train, y_train)
    else:
        # fallback
        rank, reg = 2, 0.1
        best_model = CPRegressor(weight_rank=rank, reg_W=reg, tol=1e-6, n_iter_max=200, verbose=0)
        best_model.fit(X_train, y_train)
        best_hyper = {'rank': rank, 'regularization': reg}

    return best_hyper, best_model

# -------------------------
# 50 次重复
# -------------------------
all_f1 = []
all_best_rank = []
all_best_reg = []

for run_idx in range(frun):
    print(f"Run {run_idx+1}/{frun}")
    
    train_idx = train_indices_list.iloc[:, run_idx].to_numpy() - 1
    test_idx  = test_indices_list.iloc[:, run_idx].to_numpy() - 1
    
    X_train = np.transpose(EEG_x[:, :, train_idx], (2, 0, 1))  #N*64*64
    X_test  = np.transpose(EEG_x[:, :, test_idx], (2, 0, 1))
    y_train = EEG_y[train_idx]
    y_test  = EEG_y[test_idx]
    
    # 超参数选择
    rank_values = [2, 3, 4, 5]
    reg_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    best_hyper, best_model = select_cp_hyperparameters(X_train, y_train, rank_values, reg_values)
    
    # 测试集 F1
    f1 = evaluate_f1_cp(best_model, X_test, y_test)
    
    all_f1.append(f1)
    all_best_rank.append(best_hyper['rank'])
    all_best_reg.append(best_hyper['regularization'])
    
    print(f"Run {run_idx+1}: Best rank={best_hyper['rank']}, reg={best_hyper['regularization']}, Test F1={f1:.4f}")

# -------------------------
# 统计结果
# -------------------------
f1_mean = np.mean(all_f1)
f1_std  = np.std(all_f1)
print(f"\n50 次重复 CP 测试集 F1 均值: {f1_mean:.4f}, 标准差: {f1_std:.4f}")

# -------------------------
# 保存结果
# -------------------------
import pandas as pd
df_results = pd.DataFrame([{
    "f1_mean": f1_mean,
    "f1_std": f1_std
}])
save_path = os.path.join(data_path, "EEG_CP_F1_50runs.csv")
df_results.to_csv(save_path, index=False)
print(f"50 次重复 CP 结果已保存到 {save_path}")