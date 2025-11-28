import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import integral, smooth_penalty, RegressionNN

def MFDNN(p, resp, func_cov, num_basis, layer_sizes, domain_range, epochs, val_ratio, patience, lam1=0, lam2=0, epsilon=0.001, std_resp=True):  
    """
    Multi-dimensional Functional Deep Neural Network (MFDNN) training function - supports only 2D functional data.

    Parameters:
        p (int): Number of functional variables.
        resp (numpy.ndarray): Response variable.
        func_cov (numpy.ndarray): Functional covariate data with shape:
            - If p=1: (N, T1, T2)
            - If p>1: (p, N, T1, T2)
        num_basis (tuple): Number of basis functions for each dimension (M1, M2).
        layer_sizes (list): Number of neurons in each hidden layer of the neural network.
        domain_range (list): List of p elements, each being [lower_bound, upper_bound].
        epochs (int): Number of training epochs.
        val_ratio (float, optional): Validation set ratio. Defaults to None.
        patience (int, optional): Early stopping patience. Defaults to None.
        lam (float, optional): Regularization parameter. Defaults to 0.
        epsilon (float, optional): Minimum change for early stopping. Defaults to 0.001.
        std_resp (bool, optional): Whether to standardize response variable. Defaults to True.

    Returns:
        tuple: (train_losses, validation_losses, model, l21_norm)
            - train_losses (list): Training loss values.
            - validation_losses (list): Validation loss values (if val_ratio is not None).
            - model (RegressionNN): Trained neural network model.
            - l21_norm (torch.Tensor): L21 normalization coefficients for variable selection.
    """
    # Process 2D functional data using optimized integral function
    A = integral(func_cov, num_basis, domain_range)  # Shape: (N, p, M1*M2) or (N, M1*M2)
    S = smooth_penalty(func_cov, num_basis, domain_range)
    S = torch.tensor(S, dtype=torch.float32)
    N = A.shape[0]

    # Calculate input feature dimension
    if len(A.shape) == 2:  # p=1 case
        M = A.shape[1]
        input_size = M
    else:  # p>1 case
        M = A.shape[2]
        input_size = p * M

    # Dataset splitting and preprocessing
    if val_ratio is not None:
        trainX, validationX, trainy, validationy = train_test_split(A, resp, test_size=val_ratio, random_state=42)
        trainX = torch.Tensor(trainX).float()
        trainy = torch.Tensor(trainy).view(-1, 1).float()
        validationX = torch.Tensor(validationX).float()
        validationy = torch.Tensor(validationy).view(-1, 1).float()

        if std_resp:
            trainy = (trainy - torch.mean(trainy)) / torch.std(trainy)
            validationy = (validationy - torch.mean(validationy)) / torch.std(validationy)
    else:
        trainX = torch.Tensor(A).float()
        trainy = torch.Tensor(resp).view(N, 1).float()
        if std_resp:
            trainy = (trainy - torch.mean(trainy)) / torch.std(trainy)
    
    # Create and train model
    model = RegressionNN(input_size, layer_sizes) 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    validation_losses = []
    
    if patience is not None:
        best_val_loss = float('inf')
        stopping_patience = patience
    
    # Training loop
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(trainX)
        mse = criterion(outputs, trainy)
        
        if lam1 == 0 or p == 1:
            weight = model.input_weight  # shape: (n1, p * M)
            norm = torch.sum((weight @ S) * weight, dim=1)
            regularization = lam2 * torch.mean(norm)
            total_loss = mse + regularization
            total_loss.backward()
        else:
            weight = model.input_weight  # shape: (n1, p * M)
            norm1 = torch.sum(weight ** 2, dim=0).view(p, M).sum(dim=1) # p * 1
            norm2 = torch.sum((weight @ S) * weight, dim=1)
            regularization1 = lam1 * torch.sum(torch.sqrt(norm1 + 1e-6))
            regularization2 = lam2 * torch.mean(norm2)
            total_loss = mse + regularization1 + regularization2
            total_loss.backward()
        
        optimizer.step()
        train_losses.append(mse.item())
        
        # Early stopping check
        if patience is not None:
            val_loss = criterion(model(validationX), validationy)
            validation_losses.append(val_loss.item())
            
            if val_loss < best_val_loss and best_val_loss - val_loss >= epsilon:
                best_val_loss = val_loss
                stopping_patience = patience
            else:
                stopping_patience -= 1
                if stopping_patience == 0:
                    break
    
    # Calculate L21 norm for variable selection
    if p == 1:
        l21_norm = torch.sqrt(torch.sum(model.input_weight ** 2, dim=0))
    else:
        l21_norm = torch.sqrt(torch.sum(model.input_weight ** 2, dim=0).view(p, M).sum(dim=1))
    
    
    return train_losses, validation_losses, model, l21_norm

def MFDNN_predict(p, model, func_cov, num_basis, domain_range):
    """
    Predict using trained MFDNN model - supports only 2D functional data.

    Parameters:
        p (int): Number of functional variables.
        model (RegressionNN): Trained MFDNN model.
        func_cov (numpy.ndarray): Functional covariate data for test set with shape:
            - If p=1: (N, T1, T2)
            - If p>1: (p, N, T1, T2)
        num_basis (tuple): Number of basis functions for each dimension (M1, M2).
        domain_range (list): List of p elements, each being [lower_bound, upper_bound].

    Returns:
        torch.Tensor: Predicted values.
    """
    # Process 2D functional data using optimized integral function
    A = integral(func_cov, num_basis, domain_range)
    testX = torch.Tensor(A).float()
    testy_pre = model(testX)
    
    return testy_pre