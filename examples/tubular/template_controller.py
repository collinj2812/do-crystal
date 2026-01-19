import do_mpc
import numpy as np

def controller(db_model, l, t_step=10.0):
    mpc_obj = do_mpc.controller.MPC(db_model)

    # setup controller
    mpc_obj.settings.n_horizon = 10
    mpc_obj.settings.t_step = t_step
    mpc_obj.settings.store_full_solution = True
    mpc_obj.settings.use_terminal_bounds = True
    mpc_obj.settings.nlpsol_opts = {'ipopt.max_iter': 2000}
    # db_model.mpc_obj.settings.set_linear_solver('ma27')

    # cost function
    lterm = db_model.aux['set_size'] + db_model.aux['maximize_feed']  # stage cost
    mterm = db_model.aux['set_size']  # terminal cost

    mpc_obj.set_objective(lterm=lterm, mterm=mterm)
    cost_rterm = 8e6
    mpc_obj.set_rterm(F=cost_rterm * 1e1, F_J=cost_rterm)

    # scaling
    scale_x = np.array([1e-1, 1e2, 1e2, 1e-3])
    for k in range(l):
        mpc_obj.scaling['_x', f'x_k-{k}'] = scale_x

    scale_u = np.array([1e-3, 1e-2])
    for k in range(l-1):
        mpc_obj.scaling['_x', f'u_k-{k+1}'] = scale_u


    # state constraints
    mpc_obj.set_nl_cons('temperature',
                                 expr=-db_model.aux['T'] + db_model.tvp['T_constraint'],
                                 ub=0, soft_constraint=True, penalty_term_cons=1e-8)

    # input constraints
    mpc_obj.bounds['lower', '_u', 'F'] = 0.9 * 0.003
    mpc_obj.bounds['upper', '_u', 'F'] = 1.1 * 0.003
    mpc_obj.bounds['lower', '_u', 'F_J'] = 0.9 * 0.04
    mpc_obj.bounds['upper', '_u', 'F_J'] = 1.1 * 0.04

    # time varying state constraints
    tvp_template = mpc_obj.get_tvp_template()

    def tvp_fun(t_now):
        ind = t_now
        tvp_template['_tvp', :, 'T_constraint'] = 310
        tvp_template['_tvp', :, 'set_size'] = 0.00015
        return tvp_template

    mpc_obj.set_tvp_fun(tvp_fun)

    return mpc_obj
