# Bug Report: scipy.odr ODR Segmentation Fault with job=1

**Target**: `scipy.odr.ODR`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `ODR.run()` with `job=1` parameter causes a segmentation fault, crashing the Python interpreter. This is a critical memory safety issue in the underlying ODRPACK Fortran code.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.odr


@given(
    n_points=st.integers(min_value=10, max_value=30),
    job=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=1000)
)
def test_odr_job_parameter(n_points, job, seed):
    np.random.seed(seed)

    x = np.linspace(0, 10, n_points)
    y = 2.0 + 3.0 * x + np.random.normal(0, 0.1, n_points)

    data = scipy.odr.Data(x, y)
    model = scipy.odr.unilinear

    odr = scipy.odr.ODR(data, model, beta0=[0, 0], job=job)
    output = odr.run()

    assert output.beta is not None
```

**Failing input**: `job=1` (with any valid data)

## Reproducing the Bug

```python
import numpy as np
import scipy.odr

np.random.seed(0)
x = np.linspace(0, 10, 10)
y = 2.0 + 3.0 * x + np.random.normal(0, 0.1, 10)

data = scipy.odr.Data(x, y)
model = scipy.odr.unilinear

odr = scipy.odr.ODR(data, model, beta0=[0, 0], job=1)
output = odr.run()
```

Running this code causes:
```
Fatal Python error: Segmentation fault
```

## Why This Is A Bug

The ODR class accepts a `job` parameter which "tells ODRPACK what tasks to perform" according to its documentation. However, setting `job=1` causes the underlying Fortran code to crash with a segmentation fault, which should never happen for any valid API call.

The crash occurs in `/scipy/odr/_odrpack.py` line 1127 in the `run()` method, indicating a memory access violation in the compiled ODRPACK extension module.

Valid API parameters should either:
1. Work correctly
2. Raise a Python exception with a clear error message

They should never cause a segfault.

## Fix

This requires investigation of the ODRPACK Fortran wrapper code to determine:
1. What valid values for `job` are supported
2. Why `job=1` causes a memory access violation
3. Adding input validation to raise `ValueError` for unsupported job values before calling into Fortran code

A defensive fix would be to add validation in `ODR.__init__()` or `ODR.run()`:

```python
def run(self):
    # Add validation before calling Fortran code
    if self.job is not None and self.job not in [0, 2]:  # or whatever valid values are
        raise ValueError(f"Invalid job parameter: {self.job}. "
                        f"Valid values are: 0, 2")
    # ... rest of run() implementation
```

The correct fix requires consulting the ODRPACK documentation to determine all valid job values and ensuring the Fortran wrapper handles them correctly.