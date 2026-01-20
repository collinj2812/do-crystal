import pickle
import pbe_sol.PBE
from examples.aux_functions import load_PBE_param, calculate_initial_distribution, gaussian, construct_input, MPC_input, initialize_narx, narx
import yaml
import sys
import os
import do_mpc
import numpy as np

sys.path.append(os.path.join('..', '..' , 'pbe_sol'))
sys.path.append(os.path.join('..'))
from PBE import PBE

import template_model
import template_simulator
import template_controller

from aux_functions import gaussian
import cryst

# load casadi model
with open('db_model.pkl', 'rb') as f:
    casadi_model = pickle.load(f)
    l = pickle.load(f)
    no_states = pickle.load(f)
    no_inputs = pickle.load(f)

# model objects for controller and simulator
data_based_model = template_model.data_based_model(casadi_model, l, no_states, no_inputs)

# create physical model
# load parameters from yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

# setup PBE
method = 'DPBE'
coordinate = 'L'
PBE = PBE(method, coordinate)
PBE.setup(**config['pbe'], **{'kernel': cryst.constant_kernel})

# setup seed crystal distribution, use Gaussian distribution
mu = config['initial']['mu']
sigma = config['initial']['sigma']
n_init = lambda L: config['initial']['N_0']*gaussian(L, mu, sigma)
n_init_0 = n_init(PBE.L_i)

# setup do-mpc model and simulator
model = template_model.model(config['initial']['n_discr'], PBE, **config['model'], **{'state_in': n_init_0}, **{'kernel': cryst.constant_kernel})
simulator = template_simulator.simulator(model)

# setup controller
mpc = template_controller.controller(data_based_model, l)
mpc.setup()
mpc.set_initial_guess()

# initial state
x0 = simulator.x0
x0['c'] = config['initial']['c0']
x0['T'] = config['initial']['T0']
x0['T_j'] = config['initial']['T_j0']
# PBE states
x0['PBE_state'] = np.concatenate([n_init_0.reshape(-1, 1) for i in range(config['initial']['n_discr'])], axis=1)


u = simulator.u0
F_mean = 0.002
F_J_mean = 0.03

u['F'] = F_mean
u['c_in'] = 0.22
u['T_in'] = 350
u['T_j_in'] = 350
u['F_J'] = F_J_mean
u['T_env'] = 295

x_MPC_0 = MPC_input(x0.cat, PBE.L_i, config['initial']['n_discr'], config['pbe']['no_class'])
x_MPC = initialize_narx(x_MPC_0, np.array([F_mean, F_J_mean]), l)

t_steps = 40
states = []
for t_i in range(t_steps):
    u0 = mpc.make_step(x_MPC)
    u_full = construct_input(u0[0],u0[1])

    x_next = simulator.make_step(u_full)
    x_MPC_0 = MPC_input(x_next, PBE.L_i, config['initial']['n_discr'], config['pbe']['no_class'])
    x_MPC = narx(x_MPC, x_MPC_0, u_full, l)
    states.append(x_MPC_0)

states = np.array(states).squeeze()

T = states[:, 0]
T_j = states[:, 1]
c = states[:, 2]
size = states[:, 3]

F = simulator.data['_u', 'F']
F_J = simulator.data['_u', 'F_J']

import matplotlib.pyplot as plt

fig, ax = plt.subplots(6, 1, figsize=(8, 12), sharex=True)

ax[0].plot(c)
ax[0].set_ylabel('c')

ax[1].plot(T)
ax[1].set_ylabel('T')

ax[2].plot(T_j)
ax[2].set_ylabel('T_j')

ax[3].plot(size)
ax[3].set_ylabel('size at outlet')

ax[4].plot(F)
ax[4].set_ylabel('F')

ax[5].plot(F_J)
ax[5].set_ylabel('F_J')

# fig.legend()
fig.tight_layout()
fig.show()

sys.exit()