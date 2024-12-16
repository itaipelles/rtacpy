import numpy as np
from etacorpy import calc_eta_n, calc_alpha_n, calc_rta_n

def test_rta_n_vs_alpha_n():
    ns = [2**i for i in range(1,15)]
    cfs = [0.1,0.5,1,2,5,10]
    for coverage_factor in cfs:
        rta_ns = [calc_rta_n(np.arange(n), np.arange(n), coverage_factor) for n in ns]
        alpha_ns = [calc_alpha_n(n,  np.sqrt(coverage_factor/n)) for n in ns]
        assert(np.all(np.abs((np.array(rta_ns) - np.array(alpha_ns))) < 1e-10))


def test_1_on_monotone_correlations():
    ns = [2**i for i in range(1,15)]
    cfs = [0.1,0.5,1,2,5,10]
    eta_ns = [calc_eta_n(np.arange(n), np.square(np.arange(n)), cf) for n in ns for cf in cfs]
    assert(np.all(np.abs((np.array(eta_ns) - 1) < 1e-10)))
        

def test_agnostic_to_monotonic_transformations():
    ns = [3**i for i in range(1,10)]
    cfs = [0.3, 0.9, 1.7, 9.2]
    np.random.seed(42)  # Ensure reproducibility
    for n in ns:
        x = np.random.rand(n)
        y = x
        for cf in cfs:
            eta_n_1 = calc_eta_n(x,y,cf)
            x = (0.1 + np.random.rand())*x + np.random.rand()
            y = y**2 + np.random.rand()
            eta_n_2 = calc_eta_n(x,y,cf)
            assert(np.abs(eta_n_1 - eta_n_2) < 1e-10)