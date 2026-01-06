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
F_mean = 0.003
F_J_mean = 0.04

u['F'] = F_mean
u['c_in'] = 0.22
u['T_in'] = 350
u['T_j_in'] = 350
u['F_J'] = F_J_mean
u['T_env'] = 295


import matplotlib.pyplot as plt
# simulate once and plot data over time
t_steps = 50000
training_data = True
if training_data:
    training_data_y = []
    training_data_u = []
    y0 = np.array([config['initial']['c0'], config['initial']['T0'], config['initial']['T_j0'], mu])
    training_data_y.append(y0)
    u0 = np.array([F_mean, F_J_mean])
    training_data_u.append(u0)

for i in range(t_steps):
    if training_data and i > 0:  # change inputs with given probability
        # append data
        c = np.array(simulator.data['_x', 'c'][-1, -1]).reshape(-1)
        T = np.array(simulator.data['_x', 'T'][-1, -1]).reshape(-1)
        T_j = np.array(simulator.data['_x', 'T_j'][-1, -1]).reshape(-1)
        size = find_x_at_percentile(PBE.L_i, simulator.data['_x', 'PBE_state'][-1].reshape(-1, config['pbe']['no_class'])[-1], 0.5).reshape(-1, 1)[0]
        F = np.array(u['F'][-1])[0]
        F_J = np.array(u['F_J'][-1])[0]
        y = np.concatenate([c, T, T_j, size])
        u_data = np.concatenate([F, F_J])
        training_data_y.append(y)
        training_data_u.append(u_data)

        prctg = 0.1
        prob = 0.05
        if np.random.uniform(0, 1) <= prob:
            u['F'] = np.random.uniform((1-prctg) * F_mean, (1+prctg) * F_mean)
        if np.random.uniform(0, 1) <= prob:
            u['F_J'] = np.random.uniform((1-prctg) * F_J_mean, (1+prctg) * F_J_mean)

    simulator.make_step(u.cat)
    print(f'Step {i} completed.')

# save numpy array
file = 'training_data_y.npy'
np.save(file, np.array(training_data_y))
file = 'training_data_u.npy'
np.save(file, np.array(training_data_u))

t_steps_plot = 100
size = np.zeros((t_steps_plot, config['initial']['n_discr']))
for i in range(t_steps_plot):
    size[i] = np.concatenate([find_x_at_percentile(PBE.L_i, simulator.data['_x', 'PBE_state'][i].reshape(-1, config['pbe']['no_class'])[x_i], 0.5).reshape(-1, 1) for x_i in range(config['initial']['n_discr'])]).ravel()

plt.plot(size)
plt.show()


sys.exit()