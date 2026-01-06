import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import casadi as ca

# define some functions

def gaussian(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)



def find_x_at_percentile(x, y, percentile):
    y_normalized = np.cumsum(y) / np.sum(y)
    idx_below = np.max(np.where(y_normalized < percentile)[0])
    idx_above = idx_below + 1

    x0, x1 = x[idx_below], x[idx_above]
    y0, y1 = y_normalized[idx_below], y_normalized[idx_above]

    return x0 + (percentile - y0) * (x1 - x0) / (y1 - y0)

def SID(u, y, s, n, N):
    m = u.shape[1]
    l = y.shape[1]

    U_H = createBlockHankelMatrix(u, s, N)
    Y_H = createBlockHankelMatrix(y, s, N)

    # calculate projection matrix
    U_U_T_inv = np.linalg.inv(U_H@U_H.T)

    Pi = np.eye(N) - U_H.T@U_U_T_inv@U_H

    Y_Pi = Y_H@Pi

    U, S, V = np.linalg.svd(Y_Pi)

    plt.semilogy(S, 'o-')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid()
    plt.show()

    C_T = U[:l, :n]

    u1 = U[:(s-1)*l, :n]
    u2 = U[l:s*l, :n]

    A_T = np.linalg.lstsq(u1, u2)[0].T

    eigenvalues = np.linalg.eigvals(A_T)
    print("Max eigenvalue magnitude:", np.max(np.abs(eigenvalues)))

    B_T, D_T, x0_T = identify_B_and_D(A_T, C_T, u, y, N, l, n, m)

    return A_T, B_T, C_T, D_T


def identify_B_and_D(A_T, C_T, u, y, N, l, n, m):
    # Initialize the phi matrix
    phi = []

    # Construct the phi matrix
    for k in range(1, N + 1):
        phi_k = []

        # First part: identified C * A^k
        CAk = C_T @ np.linalg.matrix_power(A_T, k - 1)
        phi_k.append(CAk)

        # Second part: sum from tau=1 to k of u(tau)^T ⊗ (C * A^(k-tau))
        second_part = np.zeros((l, m * n))
        for tau in range(2, k + 1):
            u_tau = u.T[:, tau - 2].reshape(1, -1)  # Note: tau-2 for 0-based indexing
            C_A_k_tau = C_T @ np.linalg.matrix_power(A_T, k - tau)
            second_part = second_part + np.kron(u_tau, C_A_k_tau)

        phi_k.append(second_part)

        # Third part: u(k)^T ⊗ I_l
        u_k = u.T[:, k - 1].reshape(1, -1)  # Note: k-1 for 0-based indexing
        I_l = np.eye(l)
        third_part = np.kron(u_k, I_l)
        phi_k.append(third_part)

        # Concatenate phi_k horizontally and append to phi
        phi_k_concat = np.hstack(phi_k)
        phi.append(phi_k_concat)

    # Stack all phi_k vertically
    phi = np.vstack(phi)

    # Reshape y for least squares problem
    y_N = y.T[:, :N]
    y_vec = y_N.flatten('F')  # 'F' for column-major (Fortran-style) order like MATLAB

    # Solve the least squares problem phi * theta = y_vec
    theta, _, _, _ = scipy.linalg.lstsq(phi, y_vec)

    # Extract x0, B, and D from theta
    x0_T = theta[:n]
    B_T = theta[n:n + m * n].reshape((n, m), order='F')
    D_T = theta[n + m * n:].reshape((l, m), order='F')

    return B_T, D_T, x0_T

def createBlockHankelMatrix(matrix, s, N):
    output_dim = matrix.shape[1]
    H_Y = np.zeros((output_dim*s, N))
    for col in range(N):
        for row in range(s):
            H_Y[row*output_dim:(row+1)*output_dim, col] = matrix[col + row, :]

    return H_Y

def narx_data(states, inputs, l):
    # go through lists of states and inputs and construct data matrix
    X_full = []
    Y_full = []
    for list_i in range(len(states)):
        data_matrix = np.hstack((states[list_i], inputs[list_i]))

        # number of time steps
        n_time_steps = data_matrix.shape[0]

        # number of states
        n_states = states[list_i].shape[1]

        # construct X and Y
        X = []
        Y = []

        for t in range(n_time_steps - l):
            X_i = []
            U_i = []
            for i in reversed(range(l)):
                X_i.append(states[list_i][i + t, :].reshape(1, -1))
                U_i.append(inputs[list_i][i + t, :].reshape(1, -1))
            X.append(np.hstack((*X_i, *U_i)))
            Y.append(states[list_i][t + l, :n_states])

        X = np.vstack(X)
        Y = np.vstack(Y)

        X_full.append(X)
        Y_full.append(Y)

    X_full = np.vstack(X_full)
    Y_full = np.vstack(Y_full)

    return X_full, Y_full

# neural network training
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X.cpu() if isinstance(X, torch.Tensor) else X
        self.Y = Y.cpu() if isinstance(Y, torch.Tensor) else Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32) if not isinstance(self.X, torch.Tensor) else self.X[idx]
        y = torch.tensor(self.Y[idx], dtype=torch.float32) if not isinstance(self.Y, torch.Tensor) else self.Y[idx]
        return x, y

class torch_model(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(torch_model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

def torch_to_casadi_function(model: torch.nn.Module, scaler_X, scaler_Y, activation: str = "sigmoid", rnn=False):
    """
    Converts a PyTorch model to a CasADi function for evaluation.

    Args:
        model (torch.nn.Module): The PyTorch model to convert.
        approx_obj: Approximate object containing data for scaling and PCA.
        activation (str): The activation function to use ("sigmoid" or "relu"). Default is "sigmoid".

    Returns:
        casadi.Function: A CasADi function that evaluates the PyTorch model.
    """

    # Automatically extract input size from the first Linear layer
    layers = list(model.children())

    # Identify and filter out only the Linear layers (ignoring any final activation function)
    linear_layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]

    if len(linear_layers) == 0:
        raise ValueError("The model does not contain any Linear layers. Cannot create CasADi function.")

    # CasADi symbolic input
    x = ca.SX.sym('x', scaler_X.mean_.shape[0])

    # Set up CasADi symbolic variables
    input_casadi = x

    # Scale input and pca
    input_casadi = (input_casadi - scaler_X.mean_) / np.sqrt(scaler_X.var_)

    # Iterate through the Linear layers of the PyTorch model
    for i, layer in enumerate(linear_layers):
        # Extract the weights and bias from the PyTorch linear layer
        weight = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy()

        # Perform the linear transformation: y = Wx + b
        input_casadi = ca.mtimes(weight, input_casadi) + bias

        # Apply activation function unless it's the last Linear layer
        if i < len(linear_layers) - 1:
            if activation == "relu":
                input_casadi = ca.fmax(input_casadi, 0)  # ReLU activation
            elif activation == "sigmoid":
                input_casadi = 1 / (1 + ca.exp(-input_casadi))  # Sigmoid activation
            elif activation == "tanh":
                input_casadi = ca.tanh(input_casadi)
            elif activation == "gelu":
                input_casadi = 0.5 * input_casadi * (
                            1 + ca.tanh(ca.sqrt(2 / ca.pi) * (input_casadi + 0.044715 * input_casadi ** 3)))
            else:
                raise ValueError("Unsupported activation function. Use 'sigmoid', 'relu', or 'tanh'.")

    # Scale output
    # for standard scaler
    input_casadi = input_casadi * np.sqrt(scaler_Y.var_) + scaler_Y.mean_

    # Create the CasADi function
    model_casadi_function = ca.Function('model', [x], [input_casadi])

    return model_casadi_function

def narx_data_input(states, inputs, l):
    data_matrix = np.hstack((states, inputs))

    # number of time steps
    n_time_steps = data_matrix.shape[0]

    # number of states
    n_states = states.shape[1]

    # construct X and Y
    X = []
    Y = []

    for t in range(n_time_steps - l + 1):
        X_i = []
        U_i = []
        for i in reversed(range(l)):
            X_i.append(states[i + t, :].reshape(1, -1))
            U_i.append(inputs[i + t, :].reshape(1, -1))
        X.append(np.hstack((*X_i, *U_i)))

    X = np.vstack(X)
    return X