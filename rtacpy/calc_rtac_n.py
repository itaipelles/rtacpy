from math import sqrt
import numpy as np
from numba import njit
from rtacpy.calc_rta_n import calc_rta_n

@njit
def calc_alpha_n(n, L):
    q = 1.0/(n+1)
    H = 0.5*L
    if L<q:
        return n*(L**2)
    elif H<q:
        return n*(L**2) - (n-1)*(L-q)**2
    
    num_overlap1 = int(H//q)
    
    c = L - q
    c_squared = c**2
    d = sqrt(2.0*((1.0-c)**2))
    e = sqrt(2.0*c_squared)
    left_over = q - H%q
    
    return min(1, d*e + c_squared + (n - 2.0*num_overlap1)*q**2 + 2.0*left_over**2)

@njit
def calc_rtac_n(x,y,coverage_factor=1.0):
    '''Calculates \(\text{RTAC}_n(S_n, \\text{coverage_factor})\) for \(S_n:=\{(x_i, y_i)\}_{i=1}^n\).
    
    :param x: A numpy.ndarray with shape (n,).
    :param y: A numpy.ndarray with shape (n,).
    :param coverage_factor: A positive float.
    :return: \(\text{RTAC}_n(S_n, \\text{coverage_factor})\).
    :rtype: float
    '''
    n = len(x)
    edge_length = np.sqrt(coverage_factor/n)
    RTA_n = calc_rta_n(x, y, edge_length=edge_length)
    alpha_n = calc_alpha_n(n, edge_length)
    rtac_n = 1 - ((RTA_n - alpha_n)/(1-np.exp(-coverage_factor)-alpha_n))
    return rtac_n

calc_rtac = calc_rtac_n