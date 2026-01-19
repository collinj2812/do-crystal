import do_mpc
import casadi as ca


def simulator(model):
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 10.0
    }
    simulator.set_param(**params_simulator)

    tvp_template = simulator.get_tvp_template()
    def tvp_fun(t_now):
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator
