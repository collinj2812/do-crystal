'''
Define functions used for calculation of crystallization
'''
import casadi as ca


# define some agglomeration kernels
def constant_kernel(L1: float, L2: float, beta: float) -> float:
    return beta


def sum_kernel(L1: float, L2: float, beta: float) -> float:
    return (L1 + L2) * beta


# define functions for crystallization
def solubility(T: float) -> float:
    '''
    Calculate solubility for given temperature.
    '''
    return 0.11238 * ca.exp(9.0849e-3 * (T - 273.15))  # Parameters from Wohlgemuth 2012


def G(rel_S: float) -> float:
    '''
    Calculate growth rate for given relative supersaturation.
    '''
    return 5.857e-5 * rel_S ** 2 * ca.tanh(0.913 / rel_S)  # Parameters from Hohmann et al. 2018


def nucl(rel_S: float) -> float:
    '''
    Calculate nucleation rate for given relative supersaturation.
    '''
    return 1e4 * ca.exp(-3e-2 / (ca.log(rel_S+1) ** 2)) * 0 # dummy value


def beta(rel_S: float) -> float:
    '''
    Calculate agglomeration rate for given relative supersaturation.
    '''
    return 1e-8* rel_S ** 2 * ca.exp(rel_S) # used for first case study performed for tubular
