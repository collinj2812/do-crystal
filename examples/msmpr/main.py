import do_mpc
import casadi as ca
import numpy as np
import yaml

import os
import sys

from examples.aux_functions import gaussian

sys.path.append(os.path.join('..', '..' , 'pbe_sol'))
sys.path.append(os.path.join('..'))
from PBE import PBE
import cryst

import template_model
import template_simulator

from aux_functions import gaussian
import plotting

# load parameters from yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

# setup PBE
method = 'DPBE'
coordinate = 'L'
PBE = PBE(method, coordinate)
PBE.setup(**config['pbe'])

# setup seed crystal distribution, use Gaussian distribution
mu = config['model']['mu']
sigma = config['model']['sigma']
n_init = lambda L: config['model']['N_0']*gaussian(L, mu, sigma)
config['model']['kernel'] = cryst.constant_kernel

n_seed = n_init(PBE.L_i)

# setup do-mpc model and simulator
model = template_model.model(PBE, **config['model'], **{'state_in': n_seed})
simulator = template_simulator.simulator(model)

# initial state
x0 = simulator.x0
x0['c'] = config['initial']['c0']
x0['T'] = config['initial']['T0']
x0['T_j'] = config['initial']['T_j0']
# PBE states
x0['PBE_state'] = x0['PBE_state']*0


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

for k in range(t_steps):
    plt.plot(PBE.L_i, simulator.data['_x', 'PBE_state'][k,:])
# plt.plot(PBE.L_i, simulator.data['_x', 'PBE_state'][-5,:])
plt.xlim([1e-6,1e-0])
plt.xscale('log')
plt.show()

data = simulator.data['_x', 'PBE_state']*PBE.del_L_i

plotting.plot_3d_pbe_solution(data, PBE, simulator.settings.t_step, t_steps, x_max=0.01)

