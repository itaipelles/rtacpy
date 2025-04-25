import numpy as np
from numba import njit
from rtacpy.calc_rtac_n import calc_rtac_n

@njit
def create_null_dist(n, coverage_factor=1, num_samples=2000):
    '''Calculates the distribution of \(\text{RTAC}_n\) under the null hypothesis of independence, for given values of n and coverage_factor.
    
    :param n: A positive integer, the sample size for each calculation of \(\text{RTAC}_n\).
    :param coverage_factor: A positive float, the coverage factor to be used for each calculation of \(\text{RTAC}_n\).
    :param num_samples: The number of samples to be included in the null distribution.
    :return: A numpy array of shape (num_samples,) with i.i.d samples of \(\text{RTAC}_n\) under the null hypothesis.
    :rtype: numpy.ndarray
    '''
    return np.array([calc_rtac_n(np.random.rand(n), np.random.rand(n), coverage_factor) for _ in range(num_samples)])

def calc_p_value(rtac_n, null_dist):
    '''Calculates the p value of a given value of \(\text{RTAC}_n\) against a given distribution under the null hypothesis.
    
    :param rtac_n: float, \(\text{RTAC}_n\) to calculate a p value for.
    :param null_dist: A numpy.ndarray with shape (N,) with samples of \(\text{RTAC}_n\) under the null hypothesis of independence.
    :return: The p value of the given \(\text{RTAC}_n\) according to the given distribution under the null.
    :rtype: float
    '''
    return (rtac_n<null_dist).mean()

def area_coverage_independence_test(x, y, coverage_factor=1, null_dist=None):
    '''Calculates \(\text{RTAC}_n(S_n, \\text{coverage_factor})\) and its p value for the given sample \(S_n:=\{(x_i, y_i)\}_{i=1}^n\).
    
    :param x: A numpy.ndarray with shape (n,).
    :param y: A numpy.ndarray with shape (n,).
    :param coverage_factor: A positive float.
    :param null_dist: Optional. A numpy.ndarray with shape (N,) with samples of \(\text{RTAC}_n\) under the null hypothesis of independence. If not supplied, a null distribution will be computed ad-hoc.
    :return: (\(\text{RTAC}_n(S_n, \\text{coverage_factor})\), p value)
    :rtype: (float, float)
    '''
    null_dist = null_dist if null_dist is not None else create_null_dist(len(x), coverage_factor)
    rtac_n = calc_rtac_n(x,y,coverage_factor)
    return rtac_n, calc_p_value(rtac_n, null_dist)