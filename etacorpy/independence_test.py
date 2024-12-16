import numpy as np
from numba import njit
from etacorpy.calc_eta_n import calc_eta_n

@njit
def create_null_dist(n, coverage_factor=1, num_samples=2000):
    return np.array([calc_eta_n(np.random.rand(n), np.random.rand(n), coverage_factor) for _ in range(num_samples)])

def calc_p_value(eta_n, null_dist):
    return (eta_n<null_dist).mean()

def area_coverage_independence_test(x, y, coverage_factor=1, null_dist=None):
    null_dist = null_dist if null_dist is not None else create_null_dist(len(x), coverage_factor)
    eta_n = calc_eta_n(x,y,coverage_factor)
    return eta_n, calc_p_value(eta_n, null_dist)