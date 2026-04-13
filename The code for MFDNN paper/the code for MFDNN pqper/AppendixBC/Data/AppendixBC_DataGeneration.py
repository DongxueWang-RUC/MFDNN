import sys
import os
import numpy as np
import random
import torch
from pathlib import Path

# =========================================================
# Paths
# Current script assumed in:
#   /MFDNN_code/AppendixBC/Data/xxx.py
# =========================================================
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE.parent
APPENDIXBC_DIR = DATA_DIR.parent
PROJECT_ROOT = APPENDIXBC_DIR.parent
METHOD_DIR = PROJECT_ROOT / "Method"

sys.path.insert(0, str(METHOD_DIR))

from utils import simpson_2d_numba

torch.manual_seed(756)
np.random.seed(756)
random.seed(756)


###########################################################
# 1. Basis functions
###########################################################
def B1(s, t):
    return np.sin(s) * np.sin(t)

def B2(s, t):
    return np.cos(s) * np.cos(t)


###########################################################
# 2. Important true variables: X1, X2
###########################################################
def x1(s, t):
    return (
        np.random.normal(1)
        + np.random.normal(1) * np.sin(s)
        + np.random.normal(1) * np.sin(t)
    )

def x2(s, t):
    return (
        np.random.normal(1)
        + np.random.normal(1) * np.cos(s)
        + np.random.normal(1) * np.cos(t)
    )


###########################################################
# 3. Noise variable functions
###########################################################
def generate_noise_function():
    a, b, c = np.random.normal(1), np.random.uniform(0.5, 2), np.random.uniform(0.5, 2)
    d, e, f = np.random.normal(1), np.random.uniform(0.5, 2), np.random.uniform(0.5, 2)
    g = np.random.normal(0.5)

    def noise_fn(s, t):
        return (
            a * np.sin(b * s + c * t)
            + d * np.cos(e * s - f * t)
            + g * s * t
            + np.random.normal(0, 0.1)
        )

    return noise_fn


###########################################################
# 4. Main data generation function
###########################################################
def generate_data(T1, T2, n, p=6, frun=1):
    t1 = np.linspace(0, 1, T1)
    t2 = np.linspace(0, 1, T2)
    t1_grid, t2_grid = np.meshgrid(t1, t2, indexing="ij")
    grid = np.vstack((t1_grid.ravel(), t2_grid.ravel())).T

    lower = np.array([0.0, 0.0], dtype=np.float64)
    upper = np.array([1.0, 1.0], dtype=np.float64)

    def run_single():
        X_array = np.zeros((p, 2 * n, T1, T2), dtype=np.float64)
        mX = []

        # Build noise functions for this run
        noise_functions = []
        if p > 2:
            for _ in range(p - 2):
                noise_functions.append(generate_noise_function())

        # Precompute basis grids once
        B1_grid = np.array([B1(s, t) for s, t in grid], dtype=np.float64).reshape(T1, T2)
        B2_grid = np.array([B2(s, t) for s, t in grid], dtype=np.float64).reshape(T1, T2)

        for k in range(2 * n):
            X1 = np.array([x1(s, t) for s, t in grid], dtype=np.float64).reshape(T1, T2)
            X2 = np.array([x2(s, t) for s, t in grid], dtype=np.float64).reshape(T1, T2)

            X_array[0, k, :, :] = X1
            X_array[1, k, :, :] = X2

            for j in range(2, p):
                fn = noise_functions[j - 2]
                X_noise = np.array([fn(s, t) for s, t in grid], dtype=np.float64).reshape(T1, T2)
                X_array[j, k, :, :] = X_noise

            val = (
                simpson_2d_numba(X1 * B1_grid, lower, upper)
                + simpson_2d_numba(X2 * B2_grid, lower, upper)
            )
            mX.append(val)

        mX = np.array(mX, dtype=np.float64)

        y = np.sin(mX) + np.cos(2 * mX)
        y = y + np.random.normal(0, 0.3 * np.std(y), 2 * n)

        return X_array, y

    results = [run_single() for _ in range(frun)]
    Xlist, ylist = zip(*results)
    return list(Xlist), list(ylist)


###########################################################
# 5. Save helper
###########################################################
def save_dataset(output_folder, T, n, p, frun=1):
    Xlist, ylist = generate_data(T, T, n, p=p, frun=frun)

    np.save(os.path.join(output_folder, f"Xlist_T{T}_n{n}_p{p}.npy"), Xlist)
    np.save(os.path.join(output_folder, f"ylist_T{T}_n{n}_p{p}.npy"), ylist)

    print(f"Saved data for T={T}, n={n}, p={p} in {output_folder}")


###########################################################
# 6. Main program
###########################################################
def main(output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # ========================
    # Part A: vary T, fixed n=200, p=100
    # ========================
    Ts = [32, 64, 128]
    # Ts = [32]
    for T in Ts:
        save_dataset(output_folder, T=T, n=200, p=100, frun=1)

    # ========================
    # Part B: vary n, fixed T=32, p=10
    # ========================
    Ns = [800]
    for n in Ns:
        save_dataset(output_folder, T=32, n=n, p=10, frun=1)


if __name__ == "__main__":
    output_folder = DATA_DIR
    main(output_folder)