import casadi as ca
import do_mpc


import sys

import pbe_sol.PBE as PBE_file
import pbe_sol.functions as functions
import pbe_sol.cryst as cryst


def model(n_discr: int, PBE: PBE_file.PBE, **kwargs) -> do_mpc.model.Model:
    model = do_mpc.model.Model('continuous')

    # model parameters
    rho = 1  # density of water
    cp = 4.186  # specific heat capacity of water
    # if 'Diff_inner' is in kwargs set diffusion coefficient of inner tube
    if 'Diff_inner' in kwargs:
        D = kwargs['Diff_inner']
    else:
        D = 1e-7  # diffusion coefficient of inner tube
    A = 1e-2  # cross sectional area of inner tube
    A_HT = 0.01  # area of heat transfer between tube and environment
    A_HT_J = 0.01  # area of heat transfer between jacket and environment
    U = 5e2  # heat transfer coefficient between tube and jacket
    L = 30   # length of tube
    V = A * L  # volume of tube
    D_J = 0  # diffusion coefficient of jacket
    A_J = 0.01  # cross sectional area of jacket
    U_J = 5e0  # heat transfer coefficient between jacket and environment
    V_J = A_J * L  # volume of jacket

    # read cryst and PBE paramters
    rho_cryst = kwargs['rho_cryst']
    kv = kwargs['kv']

    kernel = kwargs['kernel']
    if PBE.method == 'OCFE':
        state_in = kwargs['state_in'][0]
    else:
        state_in = kwargs['state_in']


    # define states and inputs
    # discrete phase
    PBE_state = model.set_variable('_x', 'PBE_state', shape=(PBE.state_shape()[0], n_discr))
    if PBE.set_alg:
        PBE_alg = model.set_variable('_z', 'PBE_alg', shape=(PBE.alg_shape()[0], n_discr))
        PBE_alg_aux = model.set_variable('_x', 'PBE_alg_aux', shape=n_discr)

    # continuous phase
    c = model.set_variable(var_type='_x', var_name='c', shape=(n_discr))
    T = model.set_variable(var_type='_x', var_name='T', shape=(n_discr))
    T_j = model.set_variable(var_type='_x', var_name='T_j', shape=(n_discr))

    F = model.set_variable(var_type='_u', var_name='F')
    c_in = model.set_variable(var_type='_u', var_name='c_in')
    T_in = model.set_variable(var_type='_u', var_name='T_in')
    T_j_in = model.set_variable(var_type='_u', var_name='T_j_in')
    F_J = model.set_variable(var_type='_u', var_name='F_J')
    T_env = model.set_variable(var_type='_u', var_name='T_env')

    ###############
    c_in = cryst.solubility(T_in)
    ###############

    # model equations
    # finite volume scheme
    dV = V / n_discr
    dV_J = V_J / n_discr
    dA = A_HT / n_discr
    dA_J = A_HT_J / n_discr

    # solubility, growth and nucleation rate for crystallization from states
    c_star = ca.SX.zeros(n_discr)  # solubility
    rel_S = ca.SX.zeros(n_discr)  # relative supersaturation
    G = ca.SX.zeros(n_discr)  # growth rate
    N = ca.SX.zeros(n_discr)  # nucleation rate
    beta = ca.SX.zeros(n_discr)  # agglomeration rate
    for i in range(n_discr):
        c_star[i] = cryst.solubility(T[i])
        rel_S[i] = c[i] / c_star[i] - 1
        G[i] = ca.fmax(0, cryst.G(rel_S[i]))
        N[i] = cryst.nucl(rel_S[i])
        beta[i] = cryst.beta(rel_S[i])

    # calculate moments of PBE for each discrete volume
    mu = ca.SX.zeros(6, n_discr)
    for i in range(n_discr):
        if PBE.set_alg:
            mu[:, i] = PBE.calc_moments(PBE_state[:, i], PBE_alg[:, i])  # fix reshaping
        else:
            mu[:, i] = PBE.calc_moments(PBE_state[:, i], 0)

    # calculate mass flow leaving continuous phase due to crystallization
    dmc_dt = ca.SX.zeros(n_discr)
    for element in range(n_discr):
        dmc_dt[element] = 3 * kv * rho_cryst * G[element] * mu[2, element] / dV / rho

    # fluxes
    flux_c = F * functions.first_order(c, c_in)
    flux_T = F * functions.first_order(T, T_in)
    flux_T_j = F_J * functions.first_order(T_j, T_j_in)

    # calculate PBE difference for finite volume scheme
    # for PBE difference must contain flux and diffusion
    # first order flux
    # PBE_flux[0,:] = ca.reshape(state_in,PBE_flux[0,:].shape)*F/dV
    # for element in range(n_discr):
    #     PBE_flux[element+1,:] = ca.reshape(PBE_state[:,element],PBE_flux[element+1,:].shape)*F/dV

    # WENO flux
    PBE_flux = ca.SX.zeros(n_discr + 1, PBE.state_shape()[0])
    for state_i in range(PBE.state_shape()[0]):  # could probably be put in vectors
        PBE_flux[:, state_i] = functions.first_order(ca.reshape(PBE_state[state_i, :], n_discr, -1),
                                               state_in[state_i]) * F / dV

    PBE_diffusion = ca.SX.zeros(n_discr, PBE.state_shape()[0])
    for state_i in range(PBE.state_shape()[0]):
        PBE_diffusion[:, state_i] = D * functions.diffusion(ca.reshape(PBE_state[state_i, :], n_discr, -1)) / dV ** 2

    PBE_diff = ca.SX.zeros(n_discr, PBE.state_shape()[0])
    for state_i in range(PBE.state_shape()[0]):  # could probably be put in vectors
        PBE_diff[:, state_i] = PBE_flux[:-1, state_i] - PBE_flux[1:, state_i] + PBE_diffusion[:, state_i]

    # diffusion
    diffusion_c = D * functions.diffusion(c) / dV ** 2
    diffusion_T = D * functions.diffusion(T) / dV ** 2
    diffusion_T_j = D * functions.diffusion(T_j) / dV ** 2

    # model equations
    dot_c = ca.SX.zeros(n_discr)
    dot_T = ca.SX.zeros(n_discr)
    dot_T_j = ca.SX.zeros(n_discr)

    for i in range(n_discr):
        dot_c[i] = -(flux_c[i + 1] - flux_c[i]) / dV + diffusion_c[i] - dmc_dt[i]
        dot_T[i] = -(flux_T[i + 1] - flux_T[i]) / dV + U * dA * (T_j[i] - T[i]) / (rho * cp * dV) + diffusion_T[i]
        dot_T_j[i] = -(flux_T_j[i + 1] - flux_T_j[i]) / dV_J - U * dA * (T_j[i] - T[i]) / (rho * cp * dV_J) + \
                     diffusion_T_j[i] + U_J * dA_J * (T_env - T_j[i]) / (rho * cp * dV_J)

    # PBE
    # rhs
    dot_PBE_state = ca.SX.zeros(PBE_state.shape)
    if PBE.set_alg:
        alg_var = ca.SX.zeros(PBE_alg.shape)
        dot_alg_aux = ca.SX.zeros(PBE_alg_aux.shape)

    if PBE.set_alg:
        dot_PBE_state[:, 0] = PBE.rhs(PBE_state[:, 0], PBE_alg[:, 0], G[0], N[0], kernel, beta[0],
                                               state_diff=PBE_diff[0, :])
        alg_var[:, 0], dot_alg_aux[0] = PBE.alg(PBE_state[:, 0], PBE_alg[:, 0], PBE_alg_aux[0], G[0], N[0], kernel,
                                                beta[0])
    else:
        dot_PBE_state[:, 0] = PBE.rhs(PBE_state[:, 0], 0, G[0], N[0], kernel, beta[0],
                                                  state_diff=PBE_diff[0, :], tau=dV / F)

    # for successive elements
    for element in range(1, n_discr):
        if PBE.set_alg:
            dot_PBE_state[:, element] = PBE.rhs(PBE_state[:, element], PBE_alg[:, element], G[element],
                                                         N[element], kernel, beta[element],
                                                         state_diff=PBE_diff[element, :])
            alg_var[:, element], dot_alg_aux[element] = PBE.alg(PBE_state[:, element], PBE_alg[:, element],
                                                                PBE_alg_aux[element], G[element], N[element], kernel,
                                                                beta[element])
        else:
            dot_PBE_state[:, element] = PBE.rhs(PBE_state[:, element], 0, G[element], N[element], kernel,
                                                         beta[element],
                                                         state_diff=PBE_diff[element, :], tau=dV / F)

    model.set_expression('mu', mu)
    model.set_expression('G', G)
    model.set_expression('rel_S', rel_S)

    # set rhs
    model.set_rhs('c', dot_c)
    model.set_rhs('T', dot_T)
    model.set_rhs('T_j', dot_T_j)

    model.set_rhs('PBE_state', dot_PBE_state)
    if PBE.set_alg:
        model.set_alg('PBE_alg', alg_var)
        model.set_rhs('PBE_alg_aux', dot_alg_aux)

    model.setup()

    return model


def data_based_model(casadi_NN, l, no_states, no_inputs):
    model = do_mpc.model.Model('discrete', 'SX')

    state_list = []
    input_list = []

    for i in range(l):
        state_list.append(model.set_variable('_x', f'x_k-{i}', shape=no_states))

    for i in range(l - 1):
        input_list.append(model.set_variable('_x', f'u_k-{i + 1}', shape=no_inputs))

    # input vector
    F = model.set_variable(var_type='_u', var_name='F')
    # c_in = model.set_variable(var_type='_u', var_name='c_in')
    # T_in = model.set_variable(var_type='_u', var_name='T_in')
    # T_j_in = model.set_variable(var_type='_u', var_name='T_j_in')
    F_J = model.set_variable(var_type='_u', var_name='F_J')
    # T_env = model.set_variable(var_type='_u', var_name='T_env')

    T_constraint = model.set_variable('_tvp', 'T_constraint')
    set_size = model.set_variable('_tvp', 'set_size')

    # set inputs that are not optimized
    T_in = 350
    T_j_in = 350
    c_in = cryst.solubility(T_in)
    T_env = 295

    u_k = ca.vertcat(F, F_J)

    # define input for NN
    x = ca.vertcat(*state_list)
    u = ca.vertcat(u_k, ca.vertcat(*input_list))

    # input for NN
    input_NN = ca.vertcat(x, u)

    # output from NN
    x_next = casadi_NN(input_NN)

    # expressions to access states
    # width = model.set_expression('width', state_list[0][1])
    c = model.set_expression('c', state_list[0][0])
    T = model.set_expression('T', state_list[0][1])
    T_j = model.set_expression('T_j', state_list[0][2])
    size = model.set_expression('size', state_list[0][3])

    model.set_expression('set_size', 1e5 * (size - set_size) ** 2)
    model.set_expression('maximize_feed', -1e-2 * F)

    # define rhs
    model.set_rhs('x_k-0', x_next)

    for i in range(l - 1):
        model.set_rhs(f'x_k-{i + 1}', state_list[i])

    model.set_rhs(f'u_k-{1}', u_k)

    for i in range(l - 2):
        model.set_rhs(f'u_k-{i + 2}', input_list[i])

    model.setup()
    return model
