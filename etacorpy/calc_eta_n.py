from math import sqrt
import numpy as np
from numba import njit
from etacorpy.calc_rta_n import calc_rta_n

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
def calc_eta_n(x,y,coverage_factor=1.0):
    n = len(x)
    edge_length = np.sqrt(coverage_factor/n)
    RTA_n = calc_rta_n(x, y, edge_length=edge_length)
    alpha_n = calc_alpha_n(n, edge_length)
    eta_n = 1 - ((RTA_n - alpha_n)/(1-np.exp(-coverage_factor)-alpha_n))
    return eta_n