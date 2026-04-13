import numpy as np
import torch.nn as nn
from numba import jit
from scipy.interpolate import BSpline
from scipy.linalg import block_diag


class RegressionNN(nn.Module):
    """
    A neural network for regression tasks with multiple hidden layers.
    """
    def __init__(self, M, hidden_layer_sizes):
        super(RegressionNN, self).__init__()

        layers = []

        self.input = nn.Linear(M, hidden_layer_sizes[0])
        layers.append(self.input)
        layers.append(nn.ReLU())

        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layer_sizes[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    @property
    def input_weight(self):
        return self.input.weight


def Sin_matrix(x, nbasis, deriv=0):
    """
    Generate sine basis matrix or its second derivatives on [x[0], x[-1]].
    """
    a, b = x[0], x[-1]
    L = b - a
    z = (x - a) / L

    B = np.zeros((len(x), nbasis))
    for m in range(1, nbasis + 1):
        if deriv == 0:
            B[:, m - 1] = np.sqrt(2.0) * np.sin(m * np.pi * z)
        elif deriv == 2:
            B[:, m - 1] = np.sqrt(2.0) * (-(m * np.pi / L) ** 2) * np.sin(m * np.pi * z)
        else:
            raise ValueError("Sin basis currently supports deriv=0 or deriv=2 only.")
    return B


def B_matrix(x, nbasis, degree=3, deriv=0):
    """
    Generate B-spline basis matrix or its derivatives.
    """
    t = np.zeros(nbasis + degree + 1)
    n_inner = nbasis - degree - 1

    t[:degree + 1] = x[0]
    t[-degree - 1:] = x[-1]

    if n_inner > 0:
        t[degree + 1:degree + 1 + n_inner] = np.linspace(x[0], x[-1], n_inner + 2)[1:-1]

    B = np.zeros((len(x), nbasis))
    for i in range(nbasis):
        coef = np.zeros(nbasis)
        coef[i] = 1.0
        spline = BSpline(t, coef, degree)

        if deriv == 0:
            B[:, i] = spline(x)
        else:
            B[:, i] = spline.derivative(deriv)(x)

    return B


def basis_matrix(x, nbasis, basis_type="bspline", degree=3, deriv=0):
    """
    Unified interface for basis construction.
    """
    if basis_type == "bspline":
        return B_matrix(x, nbasis, degree=degree, deriv=deriv)
    elif basis_type == "sin":
        return Sin_matrix(x, nbasis, deriv=deriv)
    else:
        raise ValueError(f"Unknown basis_type: {basis_type}")


# =========================================================
# Numba-accelerated Simpson integration
# =========================================================
@jit(nopython=True)
def simpson_1d_numba(y, x):
    """
    1D Simpson integration on an equally spaced grid.
    If the number of points is even, the last point is dropped.
    """
    n = len(y)
    if n % 2 == 0:
        n -= 1

    if n < 3:
        return 0.0

    h = (x[n - 1] - x[0]) / (n - 1)
    result = y[0] + y[n - 1]

    for i in range(1, n - 1):
        if i % 2 == 0:
            result += 2.0 * y[i]
        else:
            result += 4.0 * y[i]

    return result * h / 3.0


@jit(nopython=True)
def simpson_2d_numba(arr, lower, upper):
    """
    2D Simpson integration on an equally spaced grid.
    """
    T1, T2 = arr.shape
    x = np.linspace(lower[0], upper[0], T1)
    y = np.linspace(lower[1], upper[1], T2)

    integral_y = np.zeros(T1)
    for i in range(T1):
        integral_y[i] = simpson_1d_numba(arr[i, :], y)

    return simpson_1d_numba(integral_y, x)


@jit(nopython=True)
def compute_integrals_numba(func_flat, phi_products, T1, T2, lower, upper):
    """
    Compute 2D integral coefficients using Numba acceleration.

    Parameters
    ----------
    func_flat : ndarray, shape (N, T1*T2)
    phi_products : ndarray, shape (T1*T2, M1*M2)
    T1, T2 : int
    lower, upper : ndarray/list-like, shape (2,)

    Returns
    -------
    A : ndarray, shape (N, M1*M2)
    """
    N, _ = func_flat.shape
    M_total = phi_products.shape[1]

    A = np.zeros((N, M_total))

    for i in range(M_total):
        for j in range(N):
            product_2d = (func_flat[j] * phi_products[:, i]).reshape(T1, T2)
            A[j, i] = simpson_2d_numba(product_2d, lower, upper)

    return A


# =========================================================
# Functional layer
# =========================================================
def integral(func_cov, num_basis, domain_range, basis_type="bspline", degree=3, standardize=True):
    """
    Compute integral transformation for 2D functional data.

    Parameters
    ----------
    func_cov : array_like
        - If p=1: (N, T1, T2)
        - If p>1: (p, N, T1, T2)
    num_basis : tuple/list
        Number of basis functions (M1, M2)
    domain_range : list
        Domain range
    basis_type : str
        "bspline" or "sin"
    degree : int
        Degree of B-spline if basis_type="bspline"
    standardize : bool
        Whether to standardize columns of coefficients.

    Returns
    -------
    A : ndarray
        - If p=1: (N, M1*M2)
        - If p>1: (N, p, M1*M2)
    """
    if len(func_cov.shape) == 3:
        return _integral_single(
            func_cov,
            num_basis,
            domain_range,
            basis_type=basis_type,
            degree=degree,
            standardize=standardize,
        )
    else:
        return _integral_multiple(
            func_cov,
            num_basis,
            domain_range,
            basis_type=basis_type,
            degree=degree,
            standardize=standardize,
        )


def _integral_single(func_cov, num_basis, domain_range, basis_type="bspline", degree=3, standardize=True):
    N, T1, T2 = func_cov.shape
    M1, M2 = num_basis
    lower, upper = domain_range

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    t1 = np.linspace(lower[0], upper[0], T1)
    t2 = np.linspace(lower[1], upper[1], T2)

    phi_t1 = basis_matrix(t1, M1, basis_type=basis_type, degree=degree, deriv=0)
    phi_t2 = basis_matrix(t2, M2, basis_type=basis_type, degree=degree, deriv=0)

    phi_products = np.einsum("im,jn->ijmn", phi_t1, phi_t2).reshape(T1 * T2, M1 * M2)
    func_flat = func_cov.reshape(N, T1 * T2)

    A = compute_integrals_numba(
        np.asarray(func_flat, dtype=np.float64),
        np.asarray(phi_products, dtype=np.float64),
        T1, T2, lower, upper
    )

    if standardize:
        mean_ = np.mean(A, axis=0)
        std_ = np.std(A, axis=0)
        std_ = np.where(std_ < 1e-8, 1.0, std_)
        A = (A - mean_) / std_

    return A


def _integral_multiple(func_cov, num_basis, domain_range, basis_type="bspline", degree=3, standardize=True):
    p, N, T1, T2 = func_cov.shape
    M1, M2 = num_basis

    A = np.zeros((N, p, M1 * M2))

    for pp in range(p):
        lower = np.asarray(domain_range[pp][0], dtype=np.float64)
        upper = np.asarray(domain_range[pp][1], dtype=np.float64)

        t1 = np.linspace(lower[0], upper[0], T1)
        t2 = np.linspace(lower[1], upper[1], T2)

        phi_t1 = basis_matrix(t1, M1, basis_type=basis_type, degree=degree, deriv=0)
        phi_t2 = basis_matrix(t2, M2, basis_type=basis_type, degree=degree, deriv=0)

        phi_products = np.einsum("im,jn->ijmn", phi_t1, phi_t2).reshape(T1 * T2, M1 * M2)
        func_flat = func_cov[pp].reshape(N, T1 * T2)

        A_pp = compute_integrals_numba(
            np.asarray(func_flat, dtype=np.float64),
            np.asarray(phi_products, dtype=np.float64),
            T1, T2, lower, upper
        )

        if standardize:
            mean_ = np.mean(A_pp, axis=0)
            std_ = np.std(A_pp, axis=0)
            std_ = np.where(std_ < 1e-8, 1.0, std_)
            A[:, pp, :] = (A_pp - mean_) / std_
        else:
            A[:, pp, :] = A_pp

    return A


# =========================================================
# Smoothness penalty
# =========================================================
def smooth_penalty(func_cov, num_basis, domain_range, basis_type="bspline", degree=3):
    """
    Compute smoothness penalty matrices for 2D functional data.

    Returns
    -------
    penalty_matrix : ndarray
        - If p=1: total_penalty matrix of shape (M1*M2, M1*M2)
        - If p>1: block diagonal matrix of shape (p*M1*M2, p*M1*M2)
    """
    if len(func_cov.shape) == 3:
        _, T1, T2 = func_cov.shape
        M1, M2 = num_basis

        lower, upper = domain_range

        t1 = np.linspace(lower[0], upper[0], T1)
        t2 = np.linspace(lower[1], upper[1], T2)

        return _compute_penalty_matrices(
            t1, t2, M1, M2, basis_type=basis_type, degree=degree
        )
    else:
        p, _, T1, T2 = func_cov.shape
        M1, M2 = num_basis

        penalty_matrices_list = []

        for pp in range(p):
            lower = domain_range[pp][0]
            upper = domain_range[pp][1]

            t1 = np.linspace(lower[0], upper[0], T1)
            t2 = np.linspace(lower[1], upper[1], T2)

            penalty_matrices = _compute_penalty_matrices(
                t1, t2, M1, M2, basis_type=basis_type, degree=degree
            )
            penalty_matrices_list.append(penalty_matrices)

        return block_diag(*penalty_matrices_list)


def _compute_penalty_matrices(t1, t2, M1, M2, basis_type="bspline", degree=3):
    """
    Compute penalty matrices using coordinate vectors.
    """
    phi_t1 = basis_matrix(t1, M1, basis_type=basis_type, degree=degree, deriv=0)
    phi_t1_d2 = basis_matrix(t1, M1, basis_type=basis_type, degree=degree, deriv=2)
    phi_t2 = basis_matrix(t2, M2, basis_type=basis_type, degree=degree, deriv=0)
    phi_t2_d2 = basis_matrix(t2, M2, basis_type=basis_type, degree=degree, deriv=2)

    P_s = np.zeros((M1, M1))
    M_s = np.zeros((M1, M1))
    P_t = np.zeros((M2, M2))
    M_t = np.zeros((M2, M2))

    for i in range(M1):
        for j in range(i, M1):
            integrand = phi_t1_d2[:, i] * phi_t1_d2[:, j]
            P_s[i, j] = np.trapz(integrand, t1)
            if i != j:
                P_s[j, i] = P_s[i, j]

            integrand = phi_t1[:, i] * phi_t1[:, j]
            M_s[i, j] = np.trapz(integrand, t1)
            if i != j:
                M_s[j, i] = M_s[i, j]

    for i in range(M2):
        for j in range(i, M2):
            integrand = phi_t2_d2[:, i] * phi_t2_d2[:, j]
            P_t[i, j] = np.trapz(integrand, t2)
            if i != j:
                P_t[j, i] = P_t[i, j]

            integrand = phi_t2[:, i] * phi_t2[:, j]
            M_t[i, j] = np.trapz(integrand, t2)
            if i != j:
                M_t[j, i] = M_t[i, j]

    total_penalty = np.kron(P_s, M_t) + np.kron(M_s, P_t)
    return total_penalty