import casadi as ca
import do_mpc
import sys

sys.path.append('pbe_sol')
import PBE
import cryst


def model(PBE: PBE.PBE, **kwargs) -> do_mpc.model.Model:
    '''
    Model for multi-stage MSMPR using PBE class.

    Arguments:
    PBE: PBE class
    no_stages: number of stages in MSMPR
    kwargs: parameters necessary for crystallization
    '''
    model = do_mpc.model.Model('continuous', 'SX')

    # read parameters from cyst_param
    rho = kwargs['rho']
    rho_cryst = kwargs['rho_cryst']
    delta_H_cryst = kwargs['delta_H_cryst']
    c_p = kwargs['c_p']
    U = kwargs['U']
    A = kwargs['A']
    V_j = kwargs['V_j']
    rho_j = kwargs['rho_j']
    c_p_j = kwargs['c_p_j']
    kv = kwargs['kv']
    V = kwargs['V']
    kernel = kwargs['kernel']
    no_stages = kwargs['no_stages']

    # state_in represents seed crystals in inlet flow
    if PBE.method == 'OCFE':
        state_in = kwargs['state_in'][0]
    else:
        state_in = kwargs['state_in']

    # define state variables for PBE from PBE class
    PBE_state = model.set_variable('_x', 'PBE_state', shape=(PBE.state_shape()[0], no_stages))
    if PBE.set_alg:
        PBE_alg = model.set_variable('_z', 'PBE_alg', shape=(PBE.alg_shape()[0], no_stages))
        PBE_alg_aux = model.set_variable('_x', 'PBE_alg_aux', shape=no_stages)

    # state variables for MSMPR continuous phase
    T = model.set_variable('_x', 'T', shape=(no_stages))
    T_j = model.set_variable('_x', 'T_j', shape=(no_stages))
    c = model.set_variable('_x', 'c', shape=(no_stages))

    # define control variables for MSMPR
    T_j_in = model.set_variable('_u', 'T_j_in', shape=(no_stages))
    F_j = model.set_variable('_u', 'F_j', shape=(no_stages))
    F_feed = model.set_variable('_u', 'F_feed')
    T_feed = model.set_variable('_u', 'T_feed')

    # tvp for MPC
    set_L43 = model.set_variable('_tvp', 'set_L43')
    max_supersat = model.set_variable('_tvp', 'max_supersat')

    # calculate parameters
    c_feed = cryst.solubility(T_feed)
    m_PM = rho * V  # mass of liquid in MSMPR
    mf_PM = F_feed * rho  # mass flow rate of liquid phase
    m_TM = rho_j * V_j  # mass of liquid in cooling jacket
    mf_TM = F_j * rho_j  # mass flow rate of cooling water

    D = F_feed / V  # dilution rate of liquid phase in MSMPR

    # solubility, growth and nucleation rate for crystallization from states
    c_star = ca.SX.zeros(no_stages)  # solubility
    rel_S = ca.SX.zeros(no_stages)  # relative supersaturation
    G = ca.SX.zeros(no_stages)  # growth rate
    N = ca.SX.zeros(no_stages)  # nucleation rate
    beta = ca.SX.zeros(no_stages)  # agglomeration rate
    for i in range(no_stages):
        c_star[i] = cryst.solubility(T[i])
        rel_S[i] = c[i] / c_star[i] - 1
        G[i] = ca.fmax(0, cryst.G(rel_S[i]))
        N[i] = cryst.nucl(rel_S[i])
        beta[i] = cryst.beta(rel_S[i])

    # calculate moments of PBE in each stage
    mu = ca.SX.zeros(6, no_stages)
    for i in range(no_stages):
        if PBE.set_alg:
            mu[:, i] = PBE.calc_moments(PBE_state[:, i], PBE_alg[:, i])
        else:
            mu[:, i] = PBE.calc_moments(PBE_state[:, i], 0)

    # calculate inflow and outflow at each stage
    PBE_diff = ca.SX.zeros(PBE_state.shape)
    PBE_diff[:, 0] = (state_in - PBE_state[:, 0]) * F_feed / V
    for stage in range(1, no_stages):
        PBE_diff[:, stage] = (PBE_state[:, stage - 1] - PBE_state[:, stage]) * F_feed / V

    # first stage
    dmc_dt = ca.SX.zeros(no_stages)  # change of mass of disperse phase
    dot_PBE_state = ca.SX.zeros(PBE_state.shape)
    dot_agg = ca.SX.zeros(PBE_state.shape)
    dot_B = ca.SX.zeros(PBE_state.shape)
    dot_D = ca.SX.zeros(PBE_state.shape)
    if PBE.set_alg:
        alg_var = ca.SX.zeros(PBE_alg.shape)
        dot_alg_aux = ca.SX.zeros(PBE_alg_aux.shape)

    dmc_dt[0] = 3 * V * kv * rho_cryst * G[0] * mu[2, 0]

    # define rhs for PBE from PBE class
    if PBE.set_alg:
        dot_PBE_state[:, 0] = PBE.rhs(PBE_state[:, 0], PBE_alg[:, 0], G[0], N[0], kernel, beta[0],
                                      state_diff=PBE_diff[:, 0], tau=1 / D)
        alg_var[:, 0], dot_alg_aux[0] = PBE.alg(PBE_state[:, 0], PBE_alg[:, 0], PBE_alg_aux[0], G[0], N[0], kernel,
                                                beta[0])
    else:
        dot_PBE_state[:, 0] = PBE.rhs(PBE_state[:, 0], 0, G[0], N[0], kernel, beta[0], state_diff=PBE_diff[:, 0],
                                      tau=1 / D)
    # successive stages
    for stage in range(1, no_stages):
        dmc_dt[stage] = 3 * V * kv * rho_cryst * G[stage] * mu[2, stage]

        # define rhs for PBE from PBE class
        if PBE.set_alg:
            dot_PBE_state[:, stage] = PBE.rhs(PBE_state[:, stage], PBE_alg[:, stage], G[stage], N[stage], kernel,
                                              beta[stage], state_diff=PBE_diff[:, stage], tau=1 / D)
            alg_var[:, stage], dot_alg_aux[stage] = PBE.alg(PBE_state[:, stage], PBE_alg[:, stage], PBE_alg_aux[stage],
                                                            G[stage], N[stage], kernel, beta[stage])
        else:
            dot_PBE_state[:, stage] = PBE.rhs(PBE_state[:, stage], 0, G[stage], N[stage], kernel, beta[stage],
                                              state_diff=PBE_diff[:, stage], tau=1 / D)

    # calculate continuous phase for each stage
    dot_c = ca.SX.zeros(no_stages)  # change of concentration of L-alanine
    dot_T = ca.SX.zeros(no_stages)  # change of temperature
    dot_T_j = ca.SX.zeros(no_stages)  # change of temperature in cooling jacket

    # first stage
    # print shapes
    dot_c[0] = 1 / m_PM * (-dmc_dt[0] + mf_PM * c_feed - mf_PM * c[0])
    dot_T[0] = 1 / (m_PM * c_p) * (-delta_H_cryst * dmc_dt[0] + mf_PM * c_p * (T_feed - T[0]) - U * A * (T[0] - T_j[0]))
    dot_T_j[0] = 1 / (m_TM * c_p_j) * (
            mf_TM[0] * c_p_j * (T_j_in[0] - T_j[0]) + U * A * (T[0] - T_j[0]))

    # successive stages
    for stage in range(1, no_stages):
        dot_c[stage] = 1 / m_PM * (-dmc_dt[stage] + mf_PM * (c[stage - 1] - c[stage]))
        dot_T[stage] = 1 / (m_PM * c_p) * (
                -delta_H_cryst * dmc_dt[stage] + mf_PM * c_p * (T[stage - 1] - T[stage]) - U * A * (
                T[stage] - T_j[stage]))
        dot_T_j[stage] = 1 / (m_TM * c_p_j) * (
                mf_TM[stage] * c_p_j * (T_j_in[stage] - T_j[stage]) + U * A * (T[stage] - T_j[stage]))

    # set expressions for characteristic parameters from moments
    model.set_expression('size', ca.reshape(mu[4, :] / mu[3, :], -1, no_stages))
    model.set_expression('width', ca.reshape(ca.sqrt(mu[5, :] / mu[3, :] - (mu[4, :] / mu[3, :]) ** 2), -1, no_stages))
    model.set_expression('rel_S', rel_S)
    model.set_expression('beta', beta)

    model.set_rhs('c', dot_c)
    model.set_rhs('T', dot_T)
    model.set_rhs('T_j', dot_T_j)

    model.set_rhs('PBE_state', dot_PBE_state)
    if PBE.set_alg:
        model.set_alg('PBE_alg', alg_var)
        model.set_rhs('PBE_alg_aux', dot_alg_aux)

    model.setup()

    return model
