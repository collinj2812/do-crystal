import do_mpc
import casadi as ca
import numpy as np
import yaml
import scipy

import os
import sys

from examples.aux_functions import gaussian

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
method = 'QMOM'
coordinate = 'L'
PBE = PBE(method, coordinate)
PBE.setup(**config['pbe'], **{'kernel': cryst.constant_kernel})

# setup seed crystal distribution, use Gaussian distribution
mu = config['model']['mu']
sigma = config['model']['sigma']
n_init = lambda L: config['model']['N_0']*gaussian(L, mu, sigma)

# compute moments of seed crystal distribution
xx_left, xx_right = np.max([0, mu-8*sigma]), mu+8*sigma
xx = np.linspace(xx_left, xx_right, 1001)
moments_seed = np.array([scipy.integrate.simpson(n_init(xx)*xx**i, x=xx) for i in range(PBE.n_moments)]).reshape(-1, 1)

# setup do-mpc model and simulator
model = template_model.model(PBE, **config['model'], **{'state_in': moments_seed}, **{'kernel': cryst.constant_kernel})
simulator = template_simulator.simulator(model)

# initial state
x0 = simulator.x0
x0['c'] = config['initial']['c0']
x0['T'] = config['initial']['T0']
x0['T_j'] = config['initial']['T_j0']
# PBE states
x0['PBE_state'] = np.concatenate([moments_seed * 1e-10 for i in range(config['model']['no_stages'])], axis=1)


u = simulator.u0
u['T_j_in'] = 295
u['F_j'] = 0.1
u['F_feed'] = 0.005
u['T_feed'] = 323.15


import matplotlib.pyplot as plt
# simulate
t_steps = 100
for i in range(t_steps):
    simulator.make_step(u.cat)
    print(f'Step {i} completed.')


plt.plot(simulator.data['_aux', 'size'])
# plt.plot(PBE.L_i, simulator.data['_x', 'PBE_state'][-5,:])
plt.show()

