import numpy as np
import sklearn
import sys
from examples.aux_functions import SID, narx_data, TimeSeriesDataset, torch_model, torch_to_casadi_function, narx_data_input
import matplotlib.pyplot as plt
import torch
import time

# load training data
training_data_y = np.load('training_data_y.npy')
training_data_u = np.load('training_data_u.npy')

# number of states
no_states = training_data_y.shape[1]

# number of inputs
no_inputs = training_data_u.shape[1]

# split in train and test data
ratio = 80  # percentage train
cut_off = int(ratio/100 * training_data_y.shape[0])
train_y = training_data_y[:cut_off]
test_y = training_data_y[cut_off:]
train_u = training_data_u[:cut_off]
test_u = training_data_u[cut_off:]

# scale data
scaler_y = sklearn.preprocessing.StandardScaler()
y_train_sc = scaler_y.fit_transform(train_y)
y_test_sc = scaler_y.fit(test_y)
scaler_u = sklearn.preprocessing.StandardScaler()
u_train_sc = scaler_u.fit_transform(train_u)
u_test_sc = scaler_u.fit(test_u)


# subspace identification
do_SID = False
if do_SID:
    N = 2500
    n = 10
    A, B, C, D = SID(u_train_sc, y_train_sc, s=10, n=n, N=N)

    # test model
    x = []
    y_sim = []
    N_sim = 500

    x0_est = np.zeros(n)
    x_0 = x0_est.reshape(-1, 1)
    x.append(x_0)

    for i in range(N_sim):
        x_next = A @ x_0 + B @ u_test_sc[i].reshape(-1, 1)
        y_next = C @ x_next + D @ u_test_sc[i].reshape(-1, 1)
        x.append(x_next)
        y_sim.append(y_next)
        x_0 = x_next

    x = np.concatenate(x, axis=1)
    y_sim = np.concatenate(y_sim, axis=1)

    plt.plot(y_sim.T, label='sim')
    plt.plot(y_test_sc[:N_sim], label='true')
    plt.legend()
    plt.show()

# NARX neural network
l = 10
X, Y = narx_data([train_y], [train_u], l=l)

X_scaler = sklearn.preprocessing.StandardScaler()
Y_scaler = sklearn.preprocessing.StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_scaled, Y_scaled, test_size=0.2)

batch_size = 64
hidden_dim = 100
learning_rate = 1e-4
weight_decay = 1e-6
n_epochs = 200

train_dataset = TimeSeriesDataset(X_train, Y_train)
val_dataset = TimeSeriesDataset(X_val, Y_val)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

input_size = X_train.shape[1]
output_size = Y_train.shape[1]

narx_nn = torch_model(input_size, output_size, hidden_dim)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(narx_nn.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=10)

train_losses = []
val_losses = []


# store best model weights
best_loss = float('inf')
best_model_weights = narx_nn.state_dict()

# training loop
for epoch in range(n_epochs):

    compute_time = 0
    data_time = 0
    narx_nn.train()
    train_loss = 0.0
    for inputs, targets in train_dataloader:
        t0 = time.time()
        t1 = time.time()
        data_time += (t1 - t0)
        # forward pass
        outputs = narx_nn(inputs)
        loss = criterion(outputs, targets)

        # backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        compute_time += (time.time() - t1)
        train_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Data: {data_time:.2f}s, Compute: {compute_time:.2f}s")

    # average train loss
    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)

    # validation loss
    narx_nn.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            outputs = narx_nn(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    # average validation loss
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)

    # save best model weights
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_weights = narx_nn.state_dict()

    # update learning rate
    scheduler.step(val_loss)

    # print losses
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

# load best model weights
narx_nn.load_state_dict(best_model_weights)

# print losses of best model
print(f'Best model, Val Loss: {best_loss}')

# plot losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

narx_nn_casadi = torch_to_casadi_function(narx_nn, X_scaler, Y_scaler, activation='gelu')

# test model
y_sim = []
N_sim = 500

y_pred_list = []
meas_array = test_y[:l]

for t_i in range(l, N_sim+l):
    # construct narx input for current step
    x = narx_data_input(meas_array[-l:], test_u[t_i - l:t_i], l)

    y_pred = narx_nn_casadi(x)

    y_pred_list.append(y_pred)
    meas_array = np.append(meas_array, y_pred.T, axis=0)

y_pred = np.array(y_pred_list).squeeze()

state = 3  # d50
plt.plot(y_pred[:, state], label='sim')
plt.plot(test_y[l:N_sim+l, state], label='true')
plt.legend()
plt.show()

sys.exit()