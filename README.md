# rtacpy

[![PyPI](https://img.shields.io/pypi/v/etacorpy.svg)](https://pypi.org/project/etacorpy/)
[![Docs](https://readthedocs.org/projects/etacorpy/badge/?version=latest)](https://etacorpy.readthedocs.io/en/latest/)
[![Tests](https://github.com/itaipelles/etacorpy/actions/workflows/test.yml/badge.svg)](https://github.com/itaipelles/etacorpy/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/itaipelles/etacorpy?include_prereleases&label=changelog)](https://github.com/itaipelles/etacorpy/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/itaipelles/etacorpy/blob/main/LICENSE)

A Python package for calculating RTAC, the rank-transform area coverage coefficient of correlation. This is the official repository of the paper (link TBD):

"A coefficient of correlation for continuous random variables based on area coverage"

## Installation

Install this library using `pip`:
```bash
pip install rtacpy
```
## Usage

```python
import numpy as np
from rtacpy import calc_rtac, create_null_dist, area_coverage_independence_test
n = 100
x = np.random.rand(n)
y = np.random.rand(n)
null_dist = create_null_dist(n)
rtac, p_value = area_coverage_independence_test(x, y, null_dist=null_dist)
print(f'x and y are independent, rtac = {rtac}, p_value = {p_value}')
y = np.square(x)
rtac, p_value = area_coverage_independence_test(x, y, null_dist=null_dist)
print(f'x and y are dependent, rtac = {rtac}, p_value = {p_value}')

# If p_value is not needed, you can calculate just rtac
rtac_2 = calc_rtac(x,y)
assert rtac == rtac_2
```

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:
```bash
cd rtacpy
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
python -m pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
