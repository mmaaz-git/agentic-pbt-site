# Bug Report: scipy.odr.ODR Segmentation Fault with job=1 and Explicit Models

**Target**: `scipy.odr.ODR`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Using `ODR.run()` with `job=1` (implicit ODR mode) while using an explicit model like `scipy.odr.unilinear` causes a segmentation fault, crashing the Python interpreter with exit code 139.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, Verbosity, example
import numpy as np
import scipy.odr


@given(
    n_points=st.integers(min_value=10, max_value=30),
    job=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=1000)
)
@example(n_points=10, job=1, seed=0)  # Force testing the failing case
@settings(verbosity=Verbosity.verbose, max_examples=1)
def test_odr_job_parameter(n_points, job, seed):
    print(f"Testing with n_points={n_points}, job={job}, seed={seed}")
    np.random.seed(seed)

    x = np.linspace(0, 10, n_points)
    y = 2.0 + 3.0 * x + np.random.normal(0, 0.1, n_points)

    data = scipy.odr.Data(x, y)
    model = scipy.odr.unilinear

    odr = scipy.odr.ODR(data, model, beta0=[0, 0], job=job)
    output = odr.run()

    assert output.beta is not None

test_odr_job_parameter()
```

<details>

<summary>
**Failing input**: `n_points=10, job=1, seed=0`
</summary>
```
Testing with n_points=10, job=1, seed=0
Falsifying example: test_odr_job_parameter(
    n_points=10,
    job=1,
    seed=0,
)
Fatal Python error: Segmentation fault

Current thread 0x00007f8d3d5a7740 (most recent call first):
  File "<stdin>", line 23 in test_odr_job_parameter
  ...
Segmentation fault (core dumped)
```
</details>

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

<details>

<summary>
Segmentation fault (exit code 139)
</summary>
```
Exit code: 139
```
</details>

## Why This Is A Bug

This is a critical bug that violates fundamental Python safety guarantees. The `job` parameter controls the fitting mode in ODRPACK, where:
- `job=0` (or job%10==0): explicit ODR fitting
- `job=1` (or job%10==1): implicit ODR fitting
- `job=2` (or job%10==2): ordinary least-squares

The scipy.odr.unilinear model is an **explicit model** (has `implicit=0`), designed for explicit ODR where y = f(x, beta). However, when `job=1` is set, it forces implicit ODR mode which expects implicit models where f(x, y, beta) ≈ 0.

The mismatch between explicit models and implicit ODR mode causes the underlying ODRPACK Fortran code to access invalid memory, resulting in a segmentation fault. The code in `/scipy/odr/_odrpack.py` at line 807 attempts to set `self.set_job(fit_type=1)` for implicit models, but there's no validation preventing users from manually setting `job=1` with explicit models.

Python libraries must **never** cause segmentation faults. Any invalid parameter combination should raise a proper Python exception (like `ValueError`) with a clear error message, not crash the interpreter. This crash:
1. Causes immediate data loss (unsaved work is lost)
2. Violates Python's memory safety guarantees
3. Could potentially be exploited as a security vulnerability
4. Makes debugging difficult for users who don't understand the internal model/job compatibility requirements

## Relevant Context

The bug occurs specifically when combining:
- An explicit model (like `unilinear`, `quadratic`, `exponential`, `multilinear` which all have `implicit=0`)
- With `job=1` or any job value where `job%10==1` (implicit ODR mode)

The scipy.odr documentation references the ODRPACK User's Guide (page 31) for job parameter details, but doesn't clearly warn about this incompatibility. The `set_job()` method documentation lists fit_type values but doesn't mention the requirement that implicit ODR requires implicit models.

Code locations:
- Crash location: `/scipy/odr/_odrpack.py:1127` in `ODR.run()`
- Model definitions: `/scipy/odr/_models.py` (unilinear defined at line 282)
- Job parameter handling: `/scipy/odr/_odrpack.py:930-1008` in `set_job()`

## Proposed Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -788,6 +788,17 @@ class ODR:
         self.output = None

         self._check()
+
+        # Validate job parameter compatibility with model type
+        if self.job is not None:
+            fit_type = self.job % 10
+            if fit_type == 1 and not self.model.implicit:
+                raise ValueError(
+                    "job parameter specifies implicit ODR (fit_type=1) "
+                    "but the model is explicit (model.implicit=0). "
+                    "Use job with fit_type=0 for explicit ODR or "
+                    "fit_type=2 for ordinary least squares with explicit models."
+                )

     def _check(self):
         """ Check the inputs for consistency, but don't bother checking things
```