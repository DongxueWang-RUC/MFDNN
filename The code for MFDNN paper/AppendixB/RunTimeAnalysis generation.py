import numpy as np
import random
import torch
import os
from utils import SimpsonIntegral2D

torch.manual_seed(756)
np.random.seed(756)
random.seed(756)

def B1(s, t): return np.sin(s) * np.sin(t)
def B2(s, t): return np.cos(s) * np.cos(t)

def x1(s, t):
    return np.random.normal(1) + np.random.normal(1) * np.sin(s) + np.random.normal(1) * np.sin(t)

def x2(s, t):
    return np.random.normal(1) + np.random.normal(1) * np.cos(s) + np.random.normal(1) * np.cos(t)

def generate_noise_function():
    # Random coefficients for noise structure
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

def generate_data(T1, T2, n, p=6, frun=1):

    t1 = np.linspace(0, 1, T1)
    t2 = np.linspace(0, 1, T2)
    t1_grid, t2_grid = np.meshgrid(t1, t2)
    grid = np.vstack((t1_grid.ravel(), t2_grid.ravel())).T

    # build noise functions if p > 2
    noise_functions = []
    if p > 2:
        for _ in range(p - 2):
            noise_functions.append(generate_noise_function())

    def run_single():
        X_array = np.zeros((p, 2 * n, T1, T2))
        mX = []

        for k in range(2 * n):

            # -------- important variables ----------
            X1 = np.array([[x1(s, t) for s, t in grid]]).reshape((T2, T1)).T
            X2 = np.array([[x2(s, t) for s, t in grid]]).reshape((T2, T1)).T

            X_array[0, k, :, :] = X1
            X_array[1, k, :, :] = X2

            # -------- noise variables ----------
            for j in range(2, p):
                fn = noise_functions[j - 2]
                X_noise = np.array([[fn(s, t) for s, t in grid]]).reshape((T2, T1)).T
                X_array[j, k, :, :] = X_noise

            # -------- compute integral mX ----------
            B1_grid = np.array([[B1(s, t) for s, t in grid]]).reshape((T2, T1)).T
            B2_grid = np.array([[B2(s, t) for s, t in grid]]).reshape((T2, T1)).T

            val = (
                SimpsonIntegral2D(X1 * B1_grid, (0, 0), (1, 1))
                + SimpsonIntegral2D(X2 * B2_grid, (0, 0), (1, 1))
            )
            mX.append(val)

        mX = np.array(mX)

        # -------- nonlinear response y ----------
        y = np.sin(mX) + np.cos(2 * mX) 
        y = y + np.random.normal(0, 0.3 * np.std(y), 2 * n)

        return X_array, y

    # only run once
    results = [run_single()]
    Xlist, ylist = zip(*results)
    return list(Xlist), list(ylist)

def main(output_folder):
    os.makedirs(output_folder, exist_ok=True)

    Ts = [32, 64, 128]
    p_max = 100
    n = 200

    for T in Ts:
        X_array, y = generate_data(T, T, n, p=p_max)

        # Save full p=100
        np.save(os.path.join(output_folder, f'Xlist_T{T}_p{p_max}.npy'), X_array)
        np.save(os.path.join(output_folder, f'ylist_T{T}_p{p_max}.npy'), y)
        print(f"Saved data for T={T}, p={p_max}")

if __name__ == "__main__":
    output_folder = "RunTimeAnalysis"
    main(output_folder)