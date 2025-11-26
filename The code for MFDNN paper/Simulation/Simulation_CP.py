import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from tensorly.regression.cp_regression import CPRegressor
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

torch.manual_seed(756)
np.random.seed(756)

warnings.filterwarnings("ignore")

# 配置参数
configurations = [
    {'T': 16, 'n': 200},
    {'T': 16, 'n': 400},
    {'T': 32, 'n': 200},
    {'T': 32, 'n': 400}
]

data_path = "/Users/wangdongxue/Documents/MFDNN/MFDNN/Simulation"


def calculate_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    y_std = np.std(y_true)
    return rmse / y_std if y_std != 0 else np.inf


def CP_regression(X_train, y_train, X_test, rank, regularization=0.01):
    X_train_r = np.transpose(X_train, (1, 0, 2, 3)) # n*p*T*T
    X_test_r = np.transpose(X_test, (1, 0, 2, 3))
    model = CPRegressor(weight_rank=rank, reg_W=regularization,
                        tol=1e-6, n_iter_max=100, verbose=0)
    model.fit(X_train_r, y_train)
    y_pred = model.predict(X_test_r)
    return y_pred, model


# -------------------------------------
# 超参数选择
# -------------------------------------
def select_tensor_hyperparameters(X_train, y_train, val_fraction=0.25, random_state=756):
    rank_values = [2, 3]
    reg_values = [0.001, 0.005, 0.01, 0.05, 0.1]

    # 划分训练集和验证集
    N = X_train.shape[1]  # N: 样本数量
    indices = np.arange(N)
    train_idx, val_idx = train_test_split(indices, test_size=val_fraction, random_state=random_state)

    X_tr = X_train[:, train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train[:, val_idx]
    y_val = y_train[val_idx]

    best_model = None
    best_nrmse = np.inf
    best_hyper = None

    for rank in rank_values:
        for reg in reg_values:
            try:
                y_pred_val, model = CP_regression(X_tr, y_tr, X_val, rank, reg)
                nrmse_val = calculate_nrmse(y_val, y_pred_val)

                if nrmse_val < best_nrmse:
                    best_nrmse = nrmse_val
                    best_hyper = {'rank': rank, 'regularization': reg}
                    best_model = model
            except:
                continue

    # fallback
    if best_model is None:
        rank = 2
        reg = 0.1
        try:
            _, best_model = CP_regression(X_train, y_train, X_train, rank, reg)
        except:
            pass
        best_hyper = {'rank': rank, 'regularization': reg}

    return best_hyper, best_model



# -------------------------------------
# 测试集评估
# -------------------------------------
def evaluate_on_test(model, X_test, y_test):
    if model is None:
        return np.inf
    try:
        X_test_r = np.transpose(X_test, (1, 0, 2, 3)) # N*p*T*T
        y_pred = model.predict(X_test_r)
        return calculate_nrmse(y_test, y_pred)
    except:
        return np.inf


# -------------------------------------
# 处理一个配置
# -------------------------------------
def _process_single_config(args):
    config, data_path = args
    T, n = config['T'], config['n']
    key = f"T{T}_n{n}"

    try:
        Xlist = np.load(os.path.join(data_path, f"Xlist_T{T}_n{n}.npy"), allow_pickle=True) # p*N*T*T
        ylist = np.load(os.path.join(data_path, f"ylist_T{T}_n{n}.npy"), allow_pickle=True)

        all_nrmse = {f"y{i+1}": [] for i in range(6)}

        for run_idx in range(50):
            X = np.array(Xlist[run_idx])
            p, N, T1, T2 = X.shape

            split = N // 2
            X_train = X[:, :split]
            X_test = X[:, split:]

            for yi in range(6):
                y_full = np.array(ylist[run_idx][yi])
                y_train = y_full[:split]
                y_test = y_full[split:]

                best_hyper, best_model = select_tensor_hyperparameters(X_train, y_train)
                nrmse = evaluate_on_test(best_model, X_test, y_test)
                all_nrmse[f"y{yi+1}"].append(nrmse)

        results = {}
        for yi in range(6):
            y_key = f"y{yi+1}"
            arr = np.array(all_nrmse[y_key])
            results[y_key] = {
                "nrmse_mean": float(np.mean(arr)),
                "nrmse_std": float(np.std(arr))
            }

        return key, results

    except Exception as e:
        print(f"处理错误 CP - T={T}, n={n}: {e}")
        err = {}
        for yi in range(6):
            err[f"y{yi+1}"] = {"nrmse_mean": np.inf, "nrmse_std": 0}
        return key, err


# -------------------------------------
# 并行运行 CP 实验
# -------------------------------------
def run_cp_parallel(n_jobs=None, data_path=data_path):
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    print(f"开始 CP 回归实验...")
    start = time.time()

    args_list = [(cfg, data_path) for cfg in configurations]

    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            results_list = list(pool.imap(_process_single_config, args_list))
    else:
        results_list = [_process_single_config(a) for a in args_list]

    results = {k: v for k, v in results_list}
    print(f"CP 完成，用时 { (time.time() - start)/60:.2f} 分钟")

    return results


# -------------------------------------
# 打印结果
# -------------------------------------
def print_results_summary(results):
    print("\nCP NRMSE 结果:")
    print(f"{'配置':<12} {'响应':<6} {'均值':>10} {'标准差':>12}")
    print("-" * 40)

    for cfg, res in results.items():
        for yk, stat in res.items():
            print(f"{cfg:<12} {yk:<6} {stat['nrmse_mean']:>10.4f} {stat['nrmse_std']:>12.4f}")


# -------------------------------------
# 主函数
# -------------------------------------
def main():
    print("启动 CP 张量回归实验...")

    start = time.time()
    n_jobs = max(1, cpu_count() - 1)

    cp_results = run_cp_parallel(n_jobs, data_path)
    print_results_summary(cp_results)

    print(f"\n实验完成，总耗时 {(time.time() - start)/60:.2f} 分钟")

    # 保存 CSV 到绝对路径
    save_dir = "/Users/wangdongxue/Documents/MFDNN/MFDNN/Simulation"
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for cfg, cfg_res in cp_results.items():
        for yk, metrics in cfg_res.items():
            rows.append({
                "method": "CP",
                "config": cfg,
                "response": yk,
                "nrmse_mean": metrics["nrmse_mean"],
                "nrmse_std": metrics["nrmse_std"]
            })

    df = pd.DataFrame(rows)
    save_path = f"{save_dir}/Simulation_CP.csv"
    df.to_csv(save_path, index=False)

    print(f"结果已保存到 {save_path}")


if __name__ == "__main__":
    main()
