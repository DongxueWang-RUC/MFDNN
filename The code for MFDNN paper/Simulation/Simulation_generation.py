import numpy as np
import random
import torch
import os
from joblib import Parallel, delayed
from utils import SimpsonIntegral2D

torch.manual_seed(756)
np.random.seed(756)
random.seed(756)


# Define B functions
def B1(s, t):
    return np.sin(s) * np.sin(t)

def B2(s, t):
    return np.cos(s) * np.cos(t)

def B3(s, t):
    return s + t

def B4(s, t):
    return np.sin(s * t)

def B5(s, t):
    return np.cos(s * t)

def B6(s, t):
    return s * t

# Define x functions
def x1(s, t):
    return np.random.normal(1) + np.random.normal(1) * np.sin(s) + np.random.normal(1) * np.sin(t)

def x2(s, t):
    return np.random.normal(1) + np.random.normal(1) * np.cos(s) + np.random.normal(1) * np.cos(t)

def x3(s, t):
    return np.random.normal(1) * np.square(s) + np.random.normal(1) * np.square(t) + np.random.normal(1) * s * t

def x4(s, t):
    return np.random.normal(1) * s * np.sin(t)

def x5(s, t):
    return np.random.normal(1) * s * np.cos(t)

def x6(s, t):
    return np.random.normal(1) * np.exp(- s * t)


# Grid generation and integration calculation function
def generate_data(T1, T2, n, p=6, frun=50):
    t1 = np.linspace(0, 1, T1)
    t2 = np.linspace(0, 1, T2)
    t1_grid, t2_grid = np.meshgrid(t1, t2)
    grid = np.vstack((t1_grid.ravel(), t2_grid.ravel())).T
    
    def run_single_simulation():
        X_array = np.zeros((p, 2 * n, T1, T2))
        mX1 = []
        mX2 = []
        mX3 = []
        
        for k in range(2 * n):
            X1 = np.array([[x1(s, t) for s, t in grid]]).reshape((T2, T1)).T
            X2 = np.array([[x2(s, t) for s, t in grid]]).reshape((T2, T1)).T
            X3 = np.array([[x3(s, t) for s, t in grid]]).reshape((T2, T1)).T
            X4 = np.array([[x4(s, t) for s, t in grid]]).reshape((T2, T1)).T
            X5 = np.array([[x5(s, t) for s, t in grid]]).reshape((T2, T1)).T
            X6 = np.array([[x6(s, t) for s, t in grid]]).reshape((T2, T1)).T

            # Evaluate B functions on the grid points
            B1_grid = np.array([[B1(s, t) for s, t in grid]]).reshape((T2, T1)).T
            B2_grid = np.array([[B2(s, t) for s, t in grid]]).reshape((T2, T1)).T
            B3_grid = np.array([[B3(s, t) for s, t in grid]]).reshape((T2, T1)).T
            B4_grid = np.array([[B4(s, t) for s, t in grid]]).reshape((T2, T1)).T
            B5_grid = np.array([[B5(s, t) for s, t in grid]]).reshape((T2, T1)).T
            B6_grid = np.array([[B6(s, t) for s, t in grid]]).reshape((T2, T1)).T

            mX1.append(SimpsonIntegral2D(X1 * B1_grid, (0, 0), (1, 1)) + SimpsonIntegral2D(X2 * B2_grid, (0, 0), (1, 1)))
            mX2.append(SimpsonIntegral2D(X2 * B2_grid, (0, 0), (1, 1)) + SimpsonIntegral2D(X5 * B5_grid, (0, 0), (1, 1)) + SimpsonIntegral2D(X6 * B6_grid, (0, 0), (1, 1)))
            mX3.append(SimpsonIntegral2D(X1 * B1_grid, (0, 0), (1, 1)) + SimpsonIntegral2D(X3 * B3_grid, (0, 0), (1, 1)) + SimpsonIntegral2D(X4 * B4_grid, (0, 0), (1, 1)) + SimpsonIntegral2D(X6 * B6_grid, (0, 0), (1, 1)))

            X_array[0, k, :, :] = X1
            X_array[1, k, :, :] = X2
            X_array[2, k, :, :] = X3
            X_array[3, k, :, :] = X4
            X_array[4, k, :, :] = X5
            X_array[5, k, :, :] = X6
        mX1 = np.array(mX1)  
        mX2 = np.array(mX2)
        mX3 = np.array(mX3)
        y1 = mX1 + np.random.normal(0, 0.3 * np.std(mX1), 2 * n)
        y2 = mX2 + np.random.normal(0, 0.3 * np.std(mX2), 2 * n)
        y3 = mX3 + np.random.normal(0, 0.3 * np.std(mX3), 2 * n)
        y4 = np.sin(mX1) + np.cos(2 * mX1)
        y5 = mX2**2 - 2 * mX2
        y6 = np.exp(-mX3**2)
        y4 = y4 + np.random.normal(0, 0.3 * np.std(y4), 2 * n)
        y5 = y5 + np.random.normal(0, 0.3 * np.std(y5), 2 * n)
        y6 = y6 + np.random.normal(0, 0.3 * np.std(y6), 2 * n)
        
        return X_array, [y1, y2, y3, y4, y5, y6]
    
    results = Parallel(n_jobs=-1)(delayed(run_single_simulation)() for _ in range(frun))
    Xlist, ylist = zip(*results)
    return list(Xlist), list(ylist)


def main(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    Ts = [16, 32]
    ns = [200, 400]
    
    for T in Ts:
        for n in ns:
            Xlist, ylist = generate_data(T, T, n)
            np.save(os.path.join(output_folder, f'Xlist_T{T}_n{n}.npy'), Xlist)
            np.save(os.path.join(output_folder, f'ylist_T{T}_n{n}.npy'), ylist)
            print(f'Saved data for T={T}, n={n} in {output_folder}')

if __name__ == "__main__":
    output_folder = "/Users/wangdongxue/Documents/MFDNN/MFDNN/Simulation"
    main(output_folder)
