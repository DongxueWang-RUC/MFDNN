import numpy as np
import torch.nn as nn
from numba import jit
from scipy.integrate import simpson
from scipy.interpolate import BSpline
from scipy.linalg import block_diag

class RegressionNN(nn.Module):
    """
    A neural network for regression tasks with multiple hidden layers.
    
    Parameters:
        M (int): Input feature dimension.
        hidden_layer_sizes (list): Number of neurons in each hidden layer.
    
    Attributes:
        input (nn.Linear): Input layer with direct access to weights.
        model (nn.Sequential): Sequential container for all layers.
    """
    def __init__(self, M, hidden_layer_sizes):
        super(RegressionNN, self).__init__()
        
        layers = []
        
        # Define input layer separately and keep reference
        self.input = nn.Linear(M, hidden_layer_sizes[0])
        layers.append(self.input)
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(hidden_layer_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, M)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Flatten input while preserving batch dimension
        x = x.view(x.size(0), -1)
        return self.model(x)
    
    @property
    def input_weight(self):
        """
        Property to access input layer weights.
        
        Returns:
            torch.Tensor: Weight matrix of input layer with shape (hidden_layer_sizes[0], M)
        """
        return self.input.weight

def B_matrix(x, nbasis, degree=3, deriv=0):
    """
    Generate B-spline basis matrix or its derivatives.
    
    Parameters:
    x : array_like, 1D
        Input data points where basis functions are evaluated.
    nbasis : int
        Number of B-spline basis functions.
    degree : int, default=3
        Degree of the B-spline (e.g., 3 for cubic splines).
    deriv : int, default=0
        Order of derivative to compute (0 for basis functions, 2 for second derivatives).
        
    Returns:
    B : ndarray, shape (len(x), nbasis)
        B-spline basis matrix or its derivative matrix where B[i, j] is the value 
        of the j-th basis function (or its derivative) at x[i].
    """
    # Create knot vector
    t = np.zeros(nbasis + degree + 1)
    n_inner = nbasis - degree - 1  # number of interior knots
    
    # Set knots (equally spaced)
    t[:degree+1] = x[0]
    t[-degree-1:] = x[-1]
    
    if n_inner > 0:
        t[degree+1:degree+1+n_inner] = np.linspace(x[0], x[-1], n_inner+2)[1:-1]
    
    # Create matrix
    B = np.zeros((len(x), nbasis))
    for i in range(nbasis):
        coef = np.zeros(nbasis)
        coef[i] = 1
        spline = BSpline(t, coef, degree)
        
        if deriv == 0:
            # Basis functions
            B[:, i] = spline(x)
        else:
            # Second derivatives
            spline_deriv = spline.derivative(deriv)
            B[:, i] = spline_deriv(x)
    
    return B

def SimpsonIntegral2D(arr, lower, upper):
    """
    Compute 2D integral using Simpson's rule with Numba acceleration.
    
    Parameters:
    arr : 2D array
        Function values to be integrated
    lower : [x_lower, y_lower]
        Lower bounds of integration
    upper : [x_upper, y_upper]
        Upper bounds of integration
        
    Returns:
    integral_result : float
        Result of the 2D integration
    """
    # 使用Numba加速的版本
    return simpson_2d_numba(arr, lower, upper)

@jit(nopython=True)
def simpson_2d_numba(arr, lower, upper):
    """
    Fast 2D Simpson integration using Numba.
    
    Parameters:
    arr : ndarray, shape (M1, M2)
        2D array to integrate
    lower, upper : list
        Integration bounds [x_lower, y_lower], [x_upper, y_upper]
        
    Returns:
    float : Integration result
    """
    M1, M2 = arr.shape
    x = np.linspace(lower[0], upper[0], M1)
    y = np.linspace(lower[1], upper[1], M2)
    
    # Integrate along y-axis first
    integral_y = np.zeros(M1)
    for i in range(M1):
        integral_y[i] = simpson_1d_numba(arr[i, :], y)
    
    # Then integrate along x-axis
    return simpson_1d_numba(integral_y, x)

@jit(nopython=True)
def simpson_1d_numba(y, x):
    """
    Fast 1D Simpson integration using Numba.
    
    Parameters:
    y : ndarray
        Function values
    x : ndarray
        Coordinate points
        
    Returns:
    float : Integration result
    """
    n = len(y)
    if n % 2 == 0:
        n -= 1  # Simpson's rule requires odd number of points
    
    h = (x[-1] - x[0]) / (n - 1)
    result = y[0] + y[n-1]
    
    for i in range(1, n-1):
        if i % 2 == 0:
            result += 2 * y[i]
        else:
            result += 4 * y[i]
    
    return result * h / 3




def integral(func_cov, num_basis, domain_range):
    """
    Compute integral transformation for 2D functional data.
    
    Parameters:
    func_cov : array_like
        Functional covariates. Shape:
        - If p=1: (N, T1, T2)
        - If p>1: (p, N, T1, T2)
    num_basis : tuple
        Number of basis functions (M1, M2)
    domain_range : list
        Domain range for functional covariates
        
    Returns:
    A : ndarray
        Transformed coefficients matrix
    """
    # Determine if single or multiple functional covariates
    if len(func_cov.shape) == 3:
        return _integral_single(func_cov, num_basis, domain_range)
    else:
        return _integral_multiple(func_cov, num_basis, domain_range)

@jit(nopython=True)
def compute_integrals_numba(func_flat, phi_products, T1, T2, lower, upper):
    """
    Compute integrals using Numba acceleration.
    
    Parameters:
    func_flat : ndarray, shape (N, T1*T2)
        Flattened functional data
    phi_products : ndarray, shape (T1*T2, M1*M2)
        Precomputed basis function products
    T1, T2 : int
        Grid dimensions
    lower, upper : list
        Integration bounds
        
    Returns:
    A : ndarray, shape (N, M1*M2)
        Integral coefficients
    """
    N, total_points = func_flat.shape
    M_total = phi_products.shape[1]
    
    A = np.zeros((N, M_total))
    
    for i in range(M_total):
        for j in range(N):
            # Reshape and compute integral
            product_2d = (func_flat[j] * phi_products[:, i]).reshape(T1, T2)
            A[j, i] = simpson_2d_numba(product_2d, lower, upper)
    
    return A

def _integral_single(func_cov, num_basis, domain_range):
    """
    Optimized version for single functional covariate.
    
    Parameters:
    func_cov : ndarray, shape (N, T1, T2)
        Single functional covariate
    num_basis : tuple
        Number of basis functions (M1, M2)
    domain_range : list
        Domain range [[x_lower, y_lower], [x_upper, y_upper]]
        
    Returns:
    A : ndarray, shape (N, M1*M2)
        Transformed coefficients
    """
    N, T1, T2 = func_cov.shape
    M1, M2 = num_basis
    lower, upper = domain_range
    
    # Generate coordinate vectors and basis matrices
    t1 = np.linspace(lower[0], upper[0], T1)
    t2 = np.linspace(lower[1], upper[1], T2)
    phi_t1 = B_matrix(t1, M1)  # T1 * M1
    phi_t2 = B_matrix(t2, M2)  # T2 * M2
    
    # Precompute all basis products using efficient einsum
    phi_products = np.einsum('im,jn->ijmn', phi_t1, phi_t2).reshape(T1*T2, M1*M2)
    
    # Reshape functional data for efficient computation
    func_flat = func_cov.reshape(N, T1*T2)
    
    # Compute integrals using Numba acceleration
    A = compute_integrals_numba(func_flat, phi_products, T1, T2, lower, upper)
    
    # Standardize
    A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
    return A

def _integral_multiple(func_cov, num_basis, domain_range):
    """
    Optimized version for multiple functional covariates.
    
    Parameters:
    func_cov : ndarray, shape (p, N, T1, T2)
        Multiple functional covariates
    num_basis : tuple
        Number of basis functions (M1, M2)
    domain_range : list
        Domain ranges for each covariate
        
    Returns:
    A : ndarray, shape (N, p, M1*M2)
        Transformed coefficients
    """
    p, N, T1, T2 = func_cov.shape
    M1, M2 = num_basis
    
    A = np.zeros((N, p, M1*M2))
    
    for pp in range(p):
        lower = domain_range[pp][0]
        upper = domain_range[pp][1]
        
        # Generate coordinate vectors and basis matrices
        t1 = np.linspace(lower[0], upper[0], T1)
        t2 = np.linspace(lower[1], upper[1], T2)
        phi_t1 = B_matrix(t1, M1)  # T1 * M1
        phi_t2 = B_matrix(t2, M2)  # T2 * M2
        
        # Precompute all basis products using efficient einsum
        phi_products = np.einsum('im,jn->ijmn', phi_t1, phi_t2).reshape(T1*T2, M1*M2)
        
        # Reshape functional data for efficient computation
        func_flat = func_cov[pp].reshape(N, T1*T2)
        
        # Compute integrals using Numba acceleration
        A_pp = compute_integrals_numba(func_flat, phi_products, T1, T2, lower, upper)
        A[:, pp, :] = A_pp
        
        # Standardize
        A[:, pp, :] = (A[:, pp, :] - np.mean(A[:, pp, :], axis=0)) / np.std(A[:, pp, :], axis=0)
    
    return A

def smooth_penalty(func_cov, num_basis, domain_range):
    """
    Compute smoothness penalty matrices for 2D functional data.
    
    Parameters:
    func_cov : array_like
        Functional covariates. Shape:
        - If p=1: (N, T1, T2)
        - If p>1: (p, N, T1, T2)
    num_basis : tuple
        Number of basis functions (M1, M2)
    domain_range : list
        Domain range for functional covariates. Shape:
        - If p=1: [[x_lower, y_lower], [x_upper, y_upper]]
        - If p>1: list of p domain ranges
        
    Returns:
    penalty_matrix : ndarray
        - If p=1: total_penalty matrix of shape (M1*M2, M1*M2)
        - If p>1: block diagonal matrix of shape (p*M1*M2, p*M1*M2)
    """
    # Determine if single or multiple functional covariates
    if len(func_cov.shape) == 3:
        # Single functional covariate: (N, T1, T2)
        _, T1, T2 = func_cov.shape
        M1, M2 = num_basis
        
        # Extract domain range
        lower, upper = domain_range
        
        # Generate coordinate vectors
        t1 = np.linspace(lower[0], upper[0], T1)
        t2 = np.linspace(lower[1], upper[1], T2)
        
        # Compute penalty matrices
        penalty_matrices = _compute_penalty_matrices(t1, t2, M1, M2)
        
        return penalty_matrices
    else:
        # Multiple functional covariates: (p, N, T1, T2)
        p, _, T1, T2 = func_cov.shape
        M1, M2 = num_basis
        
        penalty_matrices_list = []
        
        for pp in range(p):
            # Extract domain range for this covariate
            lower = domain_range[pp][0]
            upper = domain_range[pp][1]
            
            # Generate coordinate vectors
            t1 = np.linspace(lower[0], upper[0], T1)
            t2 = np.linspace(lower[1], upper[1], T2)
            
            # Compute penalty matrices for this covariate
            penalty_matrices = _compute_penalty_matrices(t1, t2, M1, M2)
            penalty_matrices_list.append(penalty_matrices)
        
        # Convert list of matrices to block diagonal matrix
        block_diag_matrix = block_diag(*penalty_matrices_list)
        
        return block_diag_matrix

def _compute_penalty_matrices(t1, t2, M1, M2):
    """
    Compute penalty matrices using coordinate vectors.
    
    Parameters:
    t1, t2 : ndarray
        Coordinate vectors for s and t directions
    M1, M2 : int
        Number of basis functions for s and t directions
        
    Returns:
    total_penalty : ndarray
        Total penalty matrix of shape (M1*M2, M1*M2)
    """
    # Compute basis functions and their second derivatives
    phi_t1 = B_matrix(t1, M1, deriv=0) # T1 * M1
    phi_t1_d2 = B_matrix(t1, M1, deriv=2) # T1 * M1
    phi_t2 = B_matrix(t2, M2, deriv=0) # T2 * M2
    phi_t2_d2 = B_matrix(t2, M2, deriv=2) # T2 * M2
    
    # Initialize matrices
    P_s = np.zeros((M1, M1))
    M_s = np.zeros((M1, M1))
    P_t = np.zeros((M2, M2))
    M_t = np.zeros((M2, M2))
    
    # Compute penalty and mass matrices for s direction
    for i in range(M1):
        for j in range(i, M1):
            # Penalty matrix (second derivatives)
            integrand = phi_t1_d2[:, i] * phi_t1_d2[:, j]
            P_s[i, j] = np.trapz(integrand, t1)
            if i != j:
                P_s[j, i] = P_s[i, j]
            
            # Mass matrix (basis functions)
            integrand = phi_t1[:, i] * phi_t1[:, j]
            M_s[i, j] = np.trapz(integrand, t1)
            if i != j:
                M_s[j, i] = M_s[i, j]
    
    # Compute penalty and mass matrices for t direction
    for i in range(M2):
        for j in range(i, M2):
            # Penalty matrix (second derivatives)
            integrand = phi_t2_d2[:, i] * phi_t2_d2[:, j]
            P_t[i, j] = np.trapz(integrand, t2)
            if i != j:
                P_t[j, i] = P_t[i, j]
            
            # Mass matrix (basis functions)
            integrand = phi_t2[:, i] * phi_t2[:, j]
            M_t[i, j] = np.trapz(integrand, t2)
            if i != j:
                M_t[j, i] = M_t[i, j]
    
    # Construct total penalty matrix
    total_penalty = np.kron(P_s, M_t) + np.kron(M_s, P_t) # (M1*M2, M1*M2)
    
    return total_penalty