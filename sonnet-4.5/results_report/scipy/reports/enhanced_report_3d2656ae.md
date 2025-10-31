# Bug Report: scipy.odr.ODR.set_iprint Missing Input Validation

**Target**: `scipy.odr.ODR.set_iprint`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ODR.set_iprint` method lacks input validation for its parameters, causing a confusing internal `ValueError` when invalid values are passed instead of providing clear user-friendly error messages.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.odr import Data, ODR, unilinear
import tempfile

def make_odr():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    data = Data(x, y)
    return ODR(data, unilinear, beta0=[1.0, 0.0])

@given(
    init=st.integers(min_value=-5, max_value=10),
    iter_param=st.integers(min_value=-5, max_value=10),
    final=st.integers(min_value=-5, max_value=10)
)
@settings(max_examples=200)
def test_set_iprint_validates_inputs(init, iter_param, final):
    odr_obj = make_odr()
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        odr_obj.rptfile = f.name

    try:
        odr_obj.set_iprint(init=init, iter=iter_param, final=final)
    except ValueError as e:
        if "is not in list" in str(e):
            raise AssertionError(f"Missing input validation: {e}")

# Run the test
test_set_iprint_validates_inputs()
```

<details>

<summary>
**Failing input**: `init=-1` (or any value outside [0, 1, 2])
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 30, in <module>
  |     test_set_iprint_validates_inputs()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 13, in test_set_iprint_validates_inputs
  |     init=st.integers(min_value=-5, max_value=10),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 24, in test_set_iprint_validates_inputs
    |     odr_obj.set_iprint(init=init, iter=iter_param, final=final)
    |     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1079, in set_iprint
    |     ip[0] = ip2arg.index(iprint_l[0:2])
    |             ~~~~~~~~~~~~^^^^^^^^^^^^^^^
    | ValueError: [-1, 0] is not in list
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 27, in test_set_iprint_validates_inputs
    |     raise AssertionError(f"Missing input validation: {e}")
    | AssertionError: Missing input validation: [-1, 0] is not in list
    | Falsifying example: test_set_iprint_validates_inputs(
    |     # The test always failed when commented parts were varied together.
    |     init=-1,
    |     iter_param=0,  # or any other generated value
    |     final=0,  # or any other generated value
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/60/hypo.py:25
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 24, in test_set_iprint_validates_inputs
    |     odr_obj.set_iprint(init=init, iter=iter_param, final=final)
    |     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1080, in set_iprint
    |     ip[1] = ip2arg.index(iprint_l[2:4])
    |             ~~~~~~~~~~~~^^^^^^^^^^^^^^^
    | ValueError: [-1, 0] is not in list
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 27, in test_set_iprint_validates_inputs
    |     raise AssertionError(f"Missing input validation: {e}")
    | AssertionError: Missing input validation: [-1, 0] is not in list
    | Falsifying example: test_set_iprint_validates_inputs(
    |     init=0,
    |     iter_param=-1,
    |     final=0,  # or any other generated value
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/60/hypo.py:25
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 24, in test_set_iprint_validates_inputs
    |     odr_obj.set_iprint(init=init, iter=iter_param, final=final)
    |     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1081, in set_iprint
    |     ip[3] = ip2arg.index(iprint_l[4:6])
    |             ~~~~~~~~~~~~^^^^^^^^^^^^^^^
    | ValueError: [-1, 0] is not in list
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 27, in test_set_iprint_validates_inputs
    |     raise AssertionError(f"Missing input validation: {e}")
    | AssertionError: Missing input validation: [-1, 0] is not in list
    | Falsifying example: test_set_iprint_validates_inputs(
    |     init=0,
    |     iter_param=0,
    |     final=-1,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/60/hypo.py:25
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.odr import Data, ODR, unilinear
import tempfile

# Create minimal ODR setup
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
odr_obj = ODR(data, unilinear, beta0=[1.0, 0.0])

# Set up a temp file for reporting
with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    odr_obj.rptfile = f.name

# This should fail with a confusing error message
# The init parameter should only accept 0, 1, or 2
# but passing 3 gives a cryptic "is not in list" error
odr_obj.set_iprint(init=3)
```

<details>

<summary>
ValueError: [3, 0] is not in list
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/repo.py", line 18, in <module>
    odr_obj.set_iprint(init=3)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1079, in set_iprint
    ip[0] = ip2arg.index(iprint_l[0:2])
            ~~~~~~~~~~~~^^^^^^^^^^^^^^^
ValueError: [3, 0] is not in list
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Documentation Contract Violation**: The docstring explicitly states at lines 1022-1025 that "The permissible values are 0, 1, and 2 representing 'no report', 'short report', and 'long report' respectively." The word "permissible" creates a contract that non-permissible values should be rejected with appropriate error handling.

2. **Poor Error Messages**: When invalid values are provided, users receive cryptic error messages like `ValueError: [3, 0] is not in list` instead of clear validation errors like `ValueError: init must be 0, 1, or 2, got 3`. This exposes internal implementation details (the `ip2arg` list structure) rather than helping users understand what went wrong.

3. **Late Failure**: The error occurs deep in the implementation at line 1079 when calling `ip2arg.index()`, rather than at the API boundary where input validation should occur. This violates the fail-fast principle.

4. **Inconsistent with SciPy Standards**: Other SciPy functions validate their inputs and provide clear error messages. This method's lack of validation is inconsistent with the library's overall quality standards.

5. **User Experience Impact**: Users debugging this error must trace through the internal implementation to understand that their parameter values are invalid, rather than immediately knowing from a clear error message.

## Relevant Context

The bug occurs in the `set_iprint` method of the `ODR` class in `/scipy/odr/_odrpack.py`. The method uses an internal lookup table `ip2arg` (defined at lines 1044-1051) that contains only valid combinations of report settings:

```python
ip2arg = [[0, 0],  # none,  none
          [1, 0],  # short, none
          [2, 0],  # long,  none
          [1, 1],  # short, short
          [2, 1],  # long,  short
          [1, 2],  # short, long
          [2, 2]]  # long,  long
```

When an invalid value is provided (e.g., `init=3`), the code at lines 1062-1073 modifies `iprint_l` with the invalid value. Then at lines 1079-1081, it tries to find this invalid combination in `ip2arg` using `.index()`, which raises the unhelpful "is not in list" error.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.set_iprint.html

## Proposed Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1033,6 +1033,20 @@ class ODR:
         If the rptfile is None, then any so_* arguments supplied will raise an
         exception.
         """
+        # Validate input parameters
+        for param_name, param_value in [('init', init), ('so_init', so_init),
+                                         ('iter', iter), ('so_iter', so_iter),
+                                         ('final', final), ('so_final', so_final)]:
+            if param_value is not None and param_value not in (0, 1, 2):
+                raise ValueError(
+                    f"{param_name} must be 0, 1, or 2, got {param_value}"
+                )
+
+        # Validate iter_step
+        if iter_step is not None and iter_step not in range(10):
+            raise ValueError(
+                f"iter_step must be between 0 and 9, got {iter_step}"
+            )
+
         if self.iprint is None:
             self.iprint = 0
```