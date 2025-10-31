# Bug Report: scipy.odr.ODR.set_iprint Crashes on Valid Input Combinations

**Target**: `scipy.odr._odrpack.ODR.set_iprint`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `set_iprint` method in scipy's ODR module crashes with a `ValueError` when called with certain documented valid parameter combinations (init=0 with so_init=1 or 2) due to missing entries in an internal lookup table.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np

@given(
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=9)
)
def test_set_iprint_doesnt_crash(init, so_init, iter, final, iter_step):
    from scipy.odr import Data, Model, ODR

    def fcn(beta, x):
        return beta[0] * x + beta[1]

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    data = Data(x, y)
    model = Model(fcn)

    odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile='test.rpt')
    odr_obj.set_iprint(init=init, so_init=so_init, iter=iter, final=final, iter_step=iter_step)

# Run the test
if __name__ == "__main__":
    test_set_iprint_doesnt_crash()
```

<details>

<summary>
**Failing input**: `init=0, so_init=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 27, in <module>
    test_set_iprint_doesnt_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 5, in test_set_iprint_doesnt_crash
    st.integers(min_value=0, max_value=2),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 23, in test_set_iprint_doesnt_crash
    odr_obj.set_iprint(init=init, so_init=so_init, iter=iter, final=final, iter_step=iter_step)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1079, in set_iprint
    ip[0] = ip2arg.index(iprint_l[0:2])
            ~~~~~~~~~~~~^^^^^^^^^^^^^^^
ValueError: [0, 1] is not in list
Falsifying example: test_set_iprint_doesnt_crash(
    # The test sometimes passed when commented parts were varied together.
    init=0,
    so_init=1,
    iter=0,  # or any other generated value
    final=0,  # or any other generated value
    iter_step=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from scipy.odr import Data, Model, ODR
import numpy as np

def fcn(beta, x):
    return beta[0] * x + beta[1]

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
model = Model(fcn)

odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile='test.rpt')
odr_obj.set_iprint(init=0, so_init=1)
```

<details>

<summary>
ValueError: [0, 1] is not in list
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/repo.py", line 13, in <module>
    odr_obj.set_iprint(init=0, so_init=1)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1079, in set_iprint
    ip[0] = ip2arg.index(iprint_l[0:2])
            ~~~~~~~~~~~~^^^^^^^^^^^^^^^
ValueError: [0, 1] is not in list
```
</details>

## Why This Is A Bug

The `set_iprint` method's documentation explicitly states that `init`, `iter`, and `final` can be 0, 1, or 2 (representing "no report", "short report", and "long report"), and that `so_init`, `so_iter`, and `so_final` can also be set to these same values for stdout output. There is no documented restriction preventing the combination of init=0 with so_init=1 or 2.

The crash occurs because the internal `ip2arg` lookup table (defined at lines 1045-1051 in _odrpack.py) only contains 7 combinations out of the mathematically possible 9 combinations (3x3 matrix). When the user sets init=0 and so_init=1, the code creates the combination [0, 1] (no file report, short stdout report) at lines 1062-1065. However, when it tries to find this combination in the `ip2arg` table at line 1079, it fails with a ValueError because [0, 1] is not in the list.

The missing combinations are:
- [0, 1]: no report to file, short report to stdout
- [0, 2]: no report to file, long report to stdout

These combinations are semantically valid - a user might reasonably want console output during development without creating report files.

## Relevant Context

The scipy.odr module implements Orthogonal Distance Regression, a statistical method for fitting models to data. The `set_iprint` method controls how computation reports are generated during the fitting process.

Key code locations:
- Method definition: `/scipy/odr/_odrpack.py:1009-1083`
- Incomplete lookup table: `/scipy/odr/_odrpack.py:1045-1051`
- Error location: `/scipy/odr/_odrpack.py:1079`

The issue affects both [0, 1] and [0, 2] combinations, though [0, 1] is more likely to be encountered as users typically want shorter console output. The same pattern affects the `iter` and `final` parameters as well, not just `init`.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.set_iprint.html

## Proposed Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1043,11 +1043,14 @@ class ODR:

         # make a list to convert iprint digits to/from argument inputs
         #                   rptfile, stdout
-        ip2arg = [[0, 0],  # none,  none
-                  [1, 0],  # short, none
-                  [2, 0],  # long,  none
-                  [1, 1],  # short, short
-                  [2, 1],  # long,  short
-                  [1, 2],  # short, long
-                  [2, 2]]  # long,  long
+        ip2arg = [[0, 0],  # none,   none
+                  [1, 0],  # short,  none
+                  [2, 0],  # long,   none
+                  [0, 1],  # none,   short
+                  [1, 1],  # short,  short
+                  [2, 1],  # long,   short
+                  [0, 2],  # none,   long
+                  [1, 2],  # short,  long
+                  [2, 2]]  # long,   long

```