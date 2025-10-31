# Bug Report: scipy.odr TypeError when using delta0 without job parameter

**Target**: `scipy.odr.ODR.run()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Using the `delta0` parameter in scipy.odr.ODR without explicitly setting the `job` parameter causes a TypeError crash due to attempting integer division on None.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.odr as odr

@given(n=st.integers(min_value=5, max_value=30))
def test_delta0_initialization(n):
    """Property: delta0 can be provided for initialization"""
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + np.random.RandomState(42).randn(n) * 0.1

    def linear_func(B, x):
        return B[0] * x + B[1]

    model = odr.Model(linear_func)
    data = odr.Data(x, y)

    delta0 = np.zeros(n)

    odr_obj = odr.ODR(data, model, beta0=[1.0, 0.0], delta0=delta0)
    output = odr_obj.run()

    assert hasattr(output, 'delta')

if __name__ == "__main__":
    test_delta0_initialization()
```

<details>

<summary>
**Failing input**: `n=5`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 25, in <module>
    test_delta0_initialization()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 6, in test_delta0_initialization
    def test_delta0_initialization(n):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 20, in test_delta0_initialization
    output = odr_obj.run()
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1100, in run
    if self.delta0 is not None and (self.job // 10000) % 10 == 0:
                                    ~~~~~~~~~^^~~~~~~
TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'
Falsifying example: test_delta0_initialization(
    n=5,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.odr as odr

n = 5
x = np.linspace(0, 10, n)
y = 2 * x + 1

def linear_func(B, x):
    return B[0] * x + B[1]

model = odr.Model(linear_func)
data = odr.Data(x, y)
delta0 = np.zeros(n)

odr_obj = odr.ODR(data, model, beta0=[1.0, 0.0], delta0=delta0)
output = odr_obj.run()
```

<details>

<summary>
TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/repo.py", line 16, in <module>
    output = odr_obj.run()
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1100, in run
    if self.delta0 is not None and (self.job // 10000) % 10 == 0:
                                    ~~~~~~~~~^^~~~~~~
TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'
```
</details>

## Why This Is A Bug

This crash violates expected behavior for several reasons:

1. **Both parameters are optional**: The scipy.odr.ODR documentation lists both `delta0` and `job` as optional parameters. Users should be able to provide one without the other.

2. **No documented dependency**: The documentation doesn't state that `job` must be specified when using `delta0`. There's no indication these parameters are interdependent.

3. **Inconsistent default handling**: The `ODR.__init__()` method sets `self.job = job` where job defaults to None, but the `run()` method assumes `self.job` is an integer when performing `self.job // 10000`.

4. **The set_job() method handles None correctly**: When `set_job()` is called, it checks if `self.job is None` and initializes it to 0 (lines 986-987 in _odrpack.py), but this method is not called automatically.

5. **Principle of least surprise violated**: Users wanting to provide initial error estimates via `delta0` shouldn't need to understand ODRPACK's internal job encoding system. The job parameter controls advanced fitting options with a 5-digit code system that most users won't need.

## Relevant Context

The bug occurs in scipy/odr/_odrpack.py at line 1100:
```python
if self.delta0 is not None and (self.job // 10000) % 10 == 0:
```

This line checks if delta0 was provided AND if the job code indicates this is not a restart (first digit of job code is 0). However, when job is None (the default), the integer division fails.

**Workaround**: Users can avoid this bug by explicitly setting `job=0`:
```python
odr_obj = odr.ODR(data, model, beta0=[1.0, 0.0], delta0=delta0, job=0)
```

The documentation recommends using the `set_job()` method for clarity, but doesn't indicate it's required when using `delta0`.

## Proposed Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1097,7 +1097,7 @@ class ODR:
                  'stpd', 'sclb', 'scld', 'work', 'iwork']

-        if self.delta0 is not None and (self.job // 10000) % 10 == 0:
+        if self.delta0 is not None and self.job is not None and (self.job // 10000) % 10 == 0:
             # delta0 provided and fit is not a restart
             self._gen_work()
```

Alternatively, ensure `self.job` has a default value of 0 instead of None during initialization, which would be consistent with the documentation stating "The default value from class initialization is for all of these options set to 0" (from set_job() docstring).