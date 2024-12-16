import numpy as np
from numba import njit
from etacorpy.calc_eta_n import calc_eta_n

@njit
def create_null_dist(n, coverage_factor=1, num_samples=2000):
    '''Calculates the distribution of \(\eta_n\) under the null hypothesis of independence, for given values of n and coverage_factor.
    
    :param n: A positive integer, the sample size for each calculation of \(\eta_n\).
    :param coverage_factor: A positive float, the coverage factor to be used for each calculation of \(\eta_n\).
    :param num_samples: The number of samples to be included in the null distribution.
    :return: A numpy array of shape (num_samples,) with i.i.d samples of \(eta_n\) under the null hypothesis.
    :rtype: numpy.ndarray
    '''
    return np.array([calc_eta_n(np.random.rand(n), np.random.rand(n), coverage_factor) for _ in range(num_samples)])

def calc_p_value(eta_n, null_dist):
    '''Calculates the p value of a given value of \(\eta_n\) against a given distribution under the null hypothesis.
    
    :param eta_n: float, \(\eta_n\) to calculate a p value for.
    :param null_dist: A numpy.ndarray with shape (N,) with samples of \(\eta_n\) under the null hypothesis of independence.
    :return: The p value of the given \(\eta_n\) according to the given distribution under the null.
    :rtype: float
    '''
    return (eta_n<null_dist).mean()

def area_coverage_independence_test(x, y, coverage_factor=1, null_dist=None):
    '''Calculates \(\eta_n(S_n, \\text{coverage_factor})\) and its p value for the given sample \(S_n:=\{(x_i, y_i)\}_{i=1}^n\).
    
    :param x: A numpy.ndarray with shape (n,).
    :param y: A numpy.ndarray with shape (n,).
    :param coverage_factor: A positive float.
    :param null_dist: Optional. A numpy.ndarray with shape (N,) with samples of \(\eta_n\) under the null hypothesis of independence. If not supplied, a null distribution will be computed ad-hoc.
    :return: (\(\eta_n(S_n, \\text{coverage_factor})\), p value)
    :rtype: (float, float)
    '''
    null_dist = null_dist if null_dist is not None else create_null_dist(len(x), coverage_factor)
    eta_n = calc_eta_n(x,y,coverage_factor)
    return eta_n, calc_p_value(eta_n, null_dist)