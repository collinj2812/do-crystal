import do_mpc
import casadi as ca
import numpy as np
import yaml
import scipy

import os
import sys

from examples.aux_functions import gaussian, find_x_at_percentile

sys.path.append(os.path.join('..', '..' , 'pbe_sol'))
sys.path.append(os.path.join('..'))
from PBE import PBE

import template_model
import template_simulator

from aux_functions import gaussian
import cryst

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

# initial state
x0 = simulator.x0
x0['c'] = config['initial']['c0']
x0['T'] = config['initial']['T0']
x0['T_j'] = config['initial']['T_j0']
# PBE states
x0['PBE_state'] = np.concatenate([n_init_0.reshape(-1, 1) for i in range(config['initial']['n_discr'])], axis=1)


u = simulator.u0
u['F'] = 0.003
u['c_in'] = 0.22
u['T_in'] = 350
u['T_j_in'] = 350
u['F_J'] = 0.04
u['T_env'] = 295


import matplotlib.pyplot as plt
# simulate once and plot data over time
t_steps = 200
for i in range(t_steps):
    simulator.make_step(u.cat)
    print(f'Step {i} completed.')

size = np.zeros((t_steps, config['initial']['n_discr']))
for i in range(t_steps):
    size[i] = np.concatenate([find_x_at_percentile(PBE.L_i, simulator.data['_x', 'PBE_state'][i].reshape(-1, 50)[x_i], 0.5).reshape(-1, 1) for x_i in range(config['initial']['n_discr'])]).ravel()

plt.plot(size)
plt.show()

# gather training data for data-based models

sys.exit()