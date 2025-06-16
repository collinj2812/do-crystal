import casadi as ca
import do_mpc
import sys

sys.path.append('pbe_sol')
import PBE
import cryst


def model(PBE: PBE.PBE, **kwargs) -> do_mpc.model.Model:
    '''
    Model for MSMPR using PBE class.

    Arguments:
    PBE: PBE class
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

    # state_in represents seed crystals in inlet flow
    if PBE.method == 'OCFE':
        state_in = kwargs['state_in'][0]
    else:
        state_in = kwargs['state_in']

    # define state variables for PBE from PBE class
    PBE_state = model.set_variable('_x', 'PBE_state', shape=(PBE.state_shape()[0]))
    if PBE.set_alg:
        PBE_alg = model.set_variable('_z', 'PBE_alg', shape=(PBE.alg_shape()[0]))
        PBE_alg_aux = model.set_variable('_x', 'PBE_alg_aux')

    # state variables for MSMPR continuous phase
    T = model.set_variable('_x', 'T')
    T_j = model.set_variable('_x', 'T_j')
    c = model.set_variable('_x', 'c')

    # define control variables for MSMPR
    T_j_in = model.set_variable('_u', 'T_j_in')
    F_j = model.set_variable('_u', 'F_j')
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
    c_star = cryst.solubility(T)
    rel_S = c / c_star - 1
    G = ca.fmax(0, cryst.G(rel_S))
    N = cryst.nucl(rel_S)
    beta = cryst.beta(rel_S)

    # calculate moments of PBE in each stage
    if PBE.set_alg:
        mu = PBE.calc_moments(PBE_state, PBE_alg)  # fix reshaping
    else:
        mu = PBE.calc_moments(PBE_state, 0)

    # calculate difference between inflow and outflow
    PBE_diff = (state_in - PBE_state) * F_feed / V

    if PBE.set_alg:
        alg_var = ca.SX.zeros(PBE_alg.shape)
        dot_alg_aux = ca.SX.zeros(PBE_alg_aux.shape)

    dmc_dt = 3 * V * kv * rho_cryst * G * mu[2]  # change of mass of disperse phase

    # define rhs for PBE from PBE class
    if PBE.set_alg:
        dot_PBE_state = PBE.rhs(PBE_state, PBE_alg, G, N, kernel, beta, state_diff=PBE_diff, tau=1 / D)
        alg_var, dot_alg_aux = PBE.alg(PBE_state, PBE_alg, PBE_alg_aux, G, N, kernel, beta)
    else:
        dot_PBE_state = PBE.rhs(PBE_state, 0, G, N, kernel, beta, state_diff=PBE_diff, tau=1 / D)

    # first stage
    # print shapes
    dot_c = 1 / m_PM * (-dmc_dt + mf_PM * c_feed - mf_PM * c)
    dot_T = 1 / (m_PM * c_p) * (-delta_H_cryst * dmc_dt + mf_PM * c_p * (T_feed - T) - U * A * (T - T_j))
    dot_T_j = 1 / (m_TM * c_p_j) * (mf_TM * c_p_j * (T_j_in - T_j) + U * A * (T - T_j))

    # set expressions for characteristic parameters from moments
    model.set_expression('size', ca.reshape(mu[4] / mu[3], -1, 1))
    model.set_expression('width', ca.reshape(ca.sqrt(mu[5] / mu[3] - (mu[4] / mu[3]) ** 2), -1, 1))
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
