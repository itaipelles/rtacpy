# etacorpy

[![PyPI](https://img.shields.io/pypi/v/etacorpy.svg)](https://pypi.org/project/etacorpy/)
[![Docs](https://readthedocs.org/projects/etacorpy/badge/?version=latest)](https://etacorpy.readthedocs.io/en/latest/)
[![Tests](https://github.com/itaipelles/etacorpy/actions/workflows/test.yml/badge.svg)](https://github.com/itaipelles/etacorpy/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/itaipelles/etacorpy?include_prereleases&label=changelog)](https://github.com/itaipelles/etacorpy/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/itaipelles/etacorpy/blob/main/LICENSE)

A Python package for calculating eta_n, the rank-transform area coverage coefficient of correlation. This is the official repository of the paper (link TBD):

"A coefficient of correlation for continuous random variables based on area coverage"

## Installation

Install this library using `pip`:
```bash
pip install etacorpy
```
## Usage

```python
import numpy as np
from etacorpy import calc_eta_n
x = np.random.rand(100)
y = np.random.rand(100)
print(f'x and y are independent, eta_n = {calc_eta_n(x,y)}')
y = np.square(x)
print(f'x and y are dependent, eta_n = {calc_eta_n(x,y)}')
```

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:
```bash
cd etacorpy
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
