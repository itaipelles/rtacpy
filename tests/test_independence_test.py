import pytest
import numpy as np
from unittest.mock import patch
from etacorpy.independence_test import create_null_dist, calc_p_value, area_coverage_independence_test

@pytest.fixture
def example_data():
    np.random.seed(42)  # Ensure reproducibility
    x = np.random.rand(100)
    y = np.random.rand(100)
    return x, y

def test_create_null_dist():
    np.random.seed(42)  # Ensure reproducibility
    n = 50
    coverage_factor = 1.5
    num_samples = 100
    null_dist = create_null_dist(n, coverage_factor, num_samples)

    assert null_dist.shape == (num_samples,)
    assert np.all(null_dist >= -1)
    assert np.all(null_dist <= 1)
    assert np.mean(null_dist) <= 0.5

def test_calc_p_value():
    null_dist = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
    eta_ns = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    expected_p_values = [1, 0.8, 0.6, 0.4, 0.2, 0.0]
    
    for (eta_n, expected_p_value) in zip(eta_ns, expected_p_values):
        p_value = calc_p_value(eta_n, null_dist)
        assert p_value == expected_p_value

def test_area_coverage_independence_test(example_data):
    x, y = example_data
    coverage_factor = 1.5

    # Run the test with a provided null distribution
    null_dist = np.random.rand(2000)
    p_value = area_coverage_independence_test(x, y, coverage_factor, null_dist)
    
    assert 0 <= p_value <= 1  # p-value should be in [0, 1]

    # Run the test without a provided null distribution (generates one internally)
    p_value = area_coverage_independence_test(x, y, coverage_factor)
    assert 0 <= p_value <= 1
