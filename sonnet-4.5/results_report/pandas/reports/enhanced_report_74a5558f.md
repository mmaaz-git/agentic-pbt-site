# Bug Report: FixedForwardWindowIndexer Raises Unclear Error When Step is Zero

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer.get_window_bounds`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer.get_window_bounds raises an unclear ZeroDivisionError from numpy's internal implementation when step=0, instead of providing a descriptive ValueError that would be consistent with the function's other parameter validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    window_size=st.integers(min_value=1, max_value=100),
    num_values=st.integers(min_value=1, max_value=100),
)
def test_fixed_forward_indexer_step_zero_should_raise(window_size, num_values):
    indexer = FixedForwardWindowIndexer(window_size=window_size)

    with pytest.raises(ValueError, match="step must be"):
        indexer.get_window_bounds(num_values=num_values, step=0)

if __name__ == "__main__":
    # Run the test and capture the failure
    test_fixed_forward_indexer_step_zero_should_raise()
```

<details>

<summary>
**Failing input**: `window_size=1, num_values=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 17, in <module>
    test_fixed_forward_indexer_step_zero_should_raise()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_fixed_forward_indexer_step_zero_should_raise
    window_size=st.integers(min_value=1, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 13, in test_fixed_forward_indexer_step_zero_should_raise
    indexer.get_window_bounds(num_values=num_values, step=0)
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py", line 340, in get_window_bounds
    start = np.arange(0, num_values, step, dtype="int64")
ZeroDivisionError: division by zero
Falsifying example: test_fixed_forward_indexer_step_zero_should_raise(
    # The test always failed when commented parts were varied together.
    window_size=1,  # or any other generated value
    num_values=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

# Create a FixedForwardWindowIndexer with window_size=5
indexer = FixedForwardWindowIndexer(window_size=5)

# Attempt to get window bounds with step=0
# This should raise a descriptive ValueError but instead raises ZeroDivisionError
try:
    start, end = indexer.get_window_bounds(num_values=10, step=0)
    print("No error raised - this is unexpected!")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"Unexpected error type {type(e).__name__}: {e}")
```

<details>

<summary>
ZeroDivisionError raised instead of descriptive ValueError
</summary>
```
ZeroDivisionError: division by zero
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Inconsistent Error Handling**: The function already validates other invalid parameters with clear ValueError messages:
   - When `center=True`: raises `ValueError("Forward-looking windows can't have center=True")`
   - When `closed` is not None: raises `ValueError("Forward-looking windows don't support setting the closed argument")`
   - But `step=0` falls through to numpy's `arange` function which raises `ZeroDivisionError`

2. **Wrong Exception Type**: A `step` of 0 is a parameter validation issue that should raise `ValueError`, not `ZeroDivisionError`. The ZeroDivisionError is an implementation detail from numpy that shouldn't be exposed to users.

3. **Unclear Error Message**: "division by zero" doesn't help users understand that the problem is with the `step` parameter. A message like "step must be a positive integer" would be much clearer.

4. **Violates Principle of Least Surprise**: Users familiar with pandas error handling would expect consistent parameter validation with descriptive error messages.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py` at line 340:
```python
start = np.arange(0, num_values, step, dtype="int64")
```

When `step=0`, numpy's `arange` function raises `ZeroDivisionError` because it cannot create a range with zero step size. This is documented numpy behavior, but pandas should validate the parameter before passing it to numpy.

The function already has a pattern for handling the `step` parameter - it defaults `None` to 1 on line 337-338:
```python
if step is None:
    step = 1
```

The fix would simply extend this pattern to validate that step is positive.

## Proposed Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -335,6 +335,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
             )
         if step is None:
             step = 1
+        if step <= 0:
+            raise ValueError("step must be a positive integer")

         start = np.arange(0, num_values, step, dtype="int64")
         end = start + self.window_size
```