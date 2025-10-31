# Bug Report: pandas.core.array_algos.putmask.putmask_without_repeat Violates Contract by Allowing Repetition with Single-Element Arrays

**Target**: `pandas.core.array_algos.putmask.putmask_without_repeat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `putmask_without_repeat` function fails to enforce its documented "exact match" requirement when the replacement array has length 1, silently repeating the single value across all masked positions instead of raising a `ValueError`.

## Property-Based Test

```python
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from pandas.core.array_algos.putmask import putmask_without_repeat


@given(
    values_size=st.integers(min_value=10, max_value=50),
    new_size=st.integers(min_value=1, max_value=9)
)
@settings(max_examples=300)
def test_putmask_without_repeat_length_mismatch_error(values_size, new_size):
    assume(new_size != values_size)

    values = np.arange(values_size)
    mask = np.ones(values_size, dtype=bool)
    new = np.arange(new_size)

    with pytest.raises(ValueError, match="cannot assign mismatch"):
        putmask_without_repeat(values, mask, new)
```

<details>

<summary>
**Failing input**: `values_size=10, new_size=1`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/43
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_putmask_without_repeat_length_mismatch_error FAILED        [100%]

=================================== FAILURES ===================================
______________ test_putmask_without_repeat_length_mismatch_error _______________

    @given(
>       values_size=st.integers(min_value=10, max_value=50),
                   ^^^
        new_size=st.integers(min_value=1, max_value=9)
    )

hypo.py:8:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

values_size = 10, new_size = 1

    @given(
        values_size=st.integers(min_value=10, max_value=50),
        new_size=st.integers(min_value=1, max_value=9)
    )
    @settings(max_examples=300)
    def test_putmask_without_repeat_length_mismatch_error(values_size, new_size):
        assume(new_size != values_size)

        values = np.arange(values_size)
        mask = np.ones(values_size, dtype=bool)
        new = np.arange(new_size)

>       with pytest.raises(ValueError, match="cannot assign mismatch"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE <class 'ValueError'>
E       Falsifying example: test_putmask_without_repeat_length_mismatch_error(
E           values_size=10,  # or any other generated value
E           new_size=1,
E       )
E       Explanation:
E           These lines were always and only run by failing examples:
E               /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/array_algos/putmask.py:94

hypo.py:19: Failed
=========================== short test summary info ============================
FAILED hypo.py::test_putmask_without_repeat_length_mismatch_error - Failed: D...
============================== 1 failed in 0.47s ===============================
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos.putmask import putmask_without_repeat

# Create test data
values = np.arange(10)
mask = np.ones(10, dtype=bool)
new = np.array([999])

print("Before putmask_without_repeat:")
print(f"  values: {values}")
print(f"  mask: {mask}")
print(f"  new: {new}")
print(f"  Length of values: {len(values)}")
print(f"  Number of True values in mask: {mask.sum()}")
print(f"  Length of new: {len(new)}")

# This should raise ValueError according to documentation
# but it doesn't when new has length 1
putmask_without_repeat(values, mask, new)

print("\nAfter putmask_without_repeat:")
print(f"  values: {values}")
print("\nExpected: ValueError('cannot assign mismatch length to masked array')")
print("Actual: No error raised, single value repeated across all positions")
```

<details>

<summary>
Output shows value 999 repeated 10 times instead of raising ValueError
</summary>
```
Before putmask_without_repeat:
  values: [0 1 2 3 4 5 6 7 8 9]
  mask: [ True  True  True  True  True  True  True  True  True  True]
  new: [999]
  Length of values: 10
  Number of True values in mask: 10
  Length of new: 1

After putmask_without_repeat:
  values: [999 999 999 999 999 999 999 999 999 999]

Expected: ValueError('cannot assign mismatch length to masked array')
Actual: No error raised, single value repeated across all positions
```
</details>

## Why This Is A Bug

The function's docstring at lines 65-67 explicitly states: "np.putmask will truncate or repeat if `new` is a listlike with len(new) != len(values). We require an exact match." This establishes a clear contract that the function should NOT allow repetition when there's a length mismatch between the replacement values and the number of masked positions.

However, the implementation contains a problematic condition at line 93:
```python
elif mask.shape[-1] == shape[-1] or shape[-1] == 1:
    np.putmask(values, mask, new)
```

When `shape[-1] == 1` (i.e., when `new` is a single-element array), the function bypasses the exact match check and calls `np.putmask` directly. Since `np.putmask` by design repeats values when the replacement array is shorter than needed, this results in the single value being repeated across all 10 masked positions. This directly contradicts the function's documented purpose of preventing repetition.

The function correctly validates the length match when `nlocs == shape[-1]` (line 84) and raises an error for other mismatches (line 96), but the special case for single-element arrays creates an inconsistency that violates the documented contract.

## Relevant Context

- **Function location**: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/array_algos/putmask.py` (lines 62-98)
- **Internal function**: This is not part of the public pandas API but is used internally by `pandas.core.internals.blocks`
- **Key variables in the bug**:
  - `nlocs` = number of True values in mask (10 in our example)
  - `shape[-1]` = length of the `new` array (1 in our example)
  - The condition `shape[-1] == 1` at line 93 allows the bypass
- **Comment at lines 86-90**: Misleadingly claims `np.place` doesn't repeat values (it does), but this doesn't affect the documented contract
- **The function name itself** (`putmask_WITHOUT_REPEAT`) emphasizes that repetition should not occur

## Proposed Fix

Remove the special case for single-element arrays from the condition at line 93 to ensure all length mismatches are properly validated:

```diff
--- a/pandas/core/array_algos/putmask.py
+++ b/pandas/core/array_algos/putmask.py
@@ -90,7 +90,7 @@ def putmask_without_repeat(
             np.place(values, mask, new)
             # i.e. values[mask] = new
-        elif mask.shape[-1] == shape[-1] or shape[-1] == 1:
+        elif mask.shape[-1] == shape[-1]:
             np.putmask(values, mask, new)
         else:
             raise ValueError("cannot assign mismatch length to masked array")
```