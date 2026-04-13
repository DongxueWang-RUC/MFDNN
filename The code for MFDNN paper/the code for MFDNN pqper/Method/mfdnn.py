import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from .utils import integral, smooth_penalty, RegressionNN

def MFDNN(
    p,
    resp,
    func_cov,
    num_basis,
    layer_sizes,
    domain_range,
    epochs,
    val_ratio,
    patience,
    lam1=0,
    lam2=0,
    epsilon=0.001,
    std_resp=True,
    basis_type="bspline",
    degree=3
):
    """
    Multi-dimensional Functional Deep Neural Network (MFDNN) training function.

    Parameters
    ----------
    p : int
        Number of functional variables.
    resp : numpy.ndarray
        Response variable.
    func_cov : numpy.ndarray
        Functional covariate data with shape:
        - If p=1: (N, T1, T2)
        - If p>1: (p, N, T1, T2)
    num_basis : tuple
        Number of basis functions for each dimension (M1, M2).
    layer_sizes : list
        Number of neurons in each hidden layer of the neural network.
    domain_range : list
        List of p elements, each being [lower_bound, upper_bound].
    epochs : int
        Number of training epochs.
    val_ratio : float or None
        Validation set ratio.
    patience : int or None
        Early stopping patience.
    lam1 : float, default=0
        Group lasso regularization parameter.
    lam2 : float, default=0
        Smoothness regularization parameter.
    epsilon : float, default=0.001
        Minimum improvement for early stopping.
    std_resp : bool, default=True
        Whether to standardize response variable.
    basis_type : str, default="bspline"
        Basis type used in the functional layer.
        Supported choices: "bspline", "sin".
    degree : int, default=3
        Degree of B-spline basis when basis_type="bspline".
        Ignored when basis_type="sin".

    Returns
    -------
    tuple
        (train_losses, validation_losses, model, l21_norm)
    """
    # Functional layer
    A = integral(
        func_cov,
        num_basis,
        domain_range,
        basis_type=basis_type,
        degree=degree,
        standardize=True
    )
    S = smooth_penalty(
        func_cov,
        num_basis,
        domain_range,
        basis_type=basis_type,
        degree=degree
    )
    S = torch.tensor(S, dtype=torch.float32)
    N = A.shape[0]

    # Input feature dimension
    if len(A.shape) == 2:   # p = 1
        M = A.shape[1]
        input_size = M
    else:                   # p > 1
        M = A.shape[2]
        input_size = p * M

    # Train-validation split
    if val_ratio is not None:
        trainX, validationX, trainy, validationy = train_test_split(
            A, resp, test_size=val_ratio, random_state=42
        )
        trainX = torch.tensor(trainX, dtype=torch.float32)
        trainy = torch.tensor(trainy, dtype=torch.float32).view(-1, 1)
        validationX = torch.tensor(validationX, dtype=torch.float32)
        validationy = torch.tensor(validationy, dtype=torch.float32).view(-1, 1)

        if std_resp:
            trainy = (trainy - torch.mean(trainy)) / torch.std(trainy)
            validationy = (validationy - torch.mean(validationy)) / torch.std(validationy)
    else:
        trainX = torch.tensor(A, dtype=torch.float32)
        trainy = torch.tensor(resp, dtype=torch.float32).view(N, 1)
        if std_resp:
            trainy = (trainy - torch.mean(trainy)) / torch.std(trainy)

    # Model
    model = RegressionNN(input_size, layer_sizes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    validation_losses = []

    if patience is not None:
        best_val_loss = float("inf")
        stopping_patience = patience

    # Training loop
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(trainX)
        mse = criterion(outputs, trainy)

        weight = model.input_weight  # shape: (n1, p*M) or (n1, M)

        if lam1 == 0 or p == 1:
            norm = torch.sum((weight @ S) * weight, dim=1)
            regularization = lam2 * torch.mean(norm)
            total_loss = mse + regularization
        else:
            norm1 = torch.sum(weight ** 2, dim=0).view(p, M).sum(dim=1)   # shape: (p,)
            norm2 = torch.sum((weight @ S) * weight, dim=1)
            regularization1 = lam1 * torch.sum(torch.sqrt(norm1 + 1e-6))
            regularization2 = lam2 * torch.mean(norm2)
            total_loss = mse + regularization1 + regularization2

        total_loss.backward()
        optimizer.step()

        train_losses.append(mse.item())

        # Early stopping
        if patience is not None:
            with torch.no_grad():
                val_loss = criterion(model(validationX), validationy)
            validation_losses.append(val_loss.item())

            if val_loss < best_val_loss and best_val_loss - val_loss >= epsilon:
                best_val_loss = val_loss
                stopping_patience = patience
            else:
                stopping_patience -= 1
                if stopping_patience == 0:
                    break

    # L21 norm for variable selection
    if p == 1:
        l21_norm = torch.sqrt(torch.sum(model.input_weight ** 2, dim=0))
    else:
        l21_norm = torch.sqrt(
            torch.sum(model.input_weight ** 2, dim=0).view(p, M).sum(dim=1)
        )

    return train_losses, validation_losses, model, l21_norm


def MFDNN_predict(
    p,
    model,
    func_cov,
    num_basis,
    domain_range,
    basis_type="bspline",
    degree=3
):
    """
    Predict using trained MFDNN model.

    Parameters
    ----------
    p : int
        Number of functional variables.
    model : RegressionNN
        Trained MFDNN model.
    func_cov : numpy.ndarray
        Functional covariate data for test set with shape:
        - If p=1: (N, T1, T2)
        - If p>1: (p, N, T1, T2)
    num_basis : tuple
        Number of basis functions for each dimension (M1, M2).
    domain_range : list
        List of p elements, each being [lower_bound, upper_bound].
    basis_type : str, default="bspline"
        Basis type used in the functional layer.
        Supported choices: "bspline", "sin".
    degree : int, default=3
        Degree of B-spline basis when basis_type="bspline".
        Ignored when basis_type="sin".

    Returns
    -------
    torch.Tensor
        Predicted values.
    """
    A = integral(
        func_cov,
        num_basis,
        domain_range,
        basis_type=basis_type,
        degree=degree,
        standardize=True
    )
    testX = torch.tensor(A, dtype=torch.float32)
    testy_pre = model(testX)
    return testy_pre