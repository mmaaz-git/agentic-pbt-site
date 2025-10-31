# Bug Report: pandas.io.parsers.readers.validate_integer min_val Constraint Bypass for Float Inputs

**Target**: `pandas.io.parsers.readers.validate_integer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_integer` function fails to enforce the `min_val` constraint when processing float inputs that represent whole numbers, while correctly enforcing it for integer inputs, creating an inconsistency where semantically identical values (-5 vs -5.0) produce different validation results.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.parsers.readers import validate_integer

@settings(max_examples=1000)
@given(st.integers(max_value=-1), st.integers(min_value=0, max_value=100))
def test_validate_integer_min_val_consistency_int_vs_float(val, min_val):
    int_raised = False
    float_raised = False

    try:
        int_result = validate_integer("test", val, min_val=min_val)
    except ValueError:
        int_raised = True

    try:
        float_result = validate_integer("test", float(val), min_val=min_val)
    except ValueError:
        float_raised = True

    if val < min_val:
        assert int_raised, f"Integer {val} should raise ValueError"
        assert float_raised, f"Float {float(val)} should raise ValueError"
        assert int_raised == float_raised, "Inconsistent behavior"

# Run the test
if __name__ == "__main__":
    test_validate_integer_min_val_consistency_int_vs_float()
```

<details>

<summary>
**Failing input**: `val=-1, min_val=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 27, in <module>
    test_validate_integer_min_val_consistency_int_vs_float()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 5, in test_validate_integer_min_val_consistency_int_vs_float
    @given(st.integers(max_value=-1), st.integers(min_value=0, max_value=100))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 22, in test_validate_integer_min_val_consistency_int_vs_float
    assert float_raised, f"Float {float(val)} should raise ValueError"
           ^^^^^^^^^^^^
AssertionError: Float -1.0 should raise ValueError
Falsifying example: test_validate_integer_min_val_consistency_int_vs_float(
    # The test always failed when commented parts were varied together.
    val=-1,  # or any other generated value
    min_val=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.parsers.readers import validate_integer

# Test case demonstrating the inconsistent behavior
try:
    result = validate_integer("test_param", -5, min_val=0)
    print(f"Integer -5: {result}")
except ValueError as e:
    print(f"Integer -5: ValueError - {e}")

try:
    result = validate_integer("test_param", -5.0, min_val=0)
    print(f"Float -5.0: {result}")
except ValueError as e:
    print(f"Float -5.0: ValueError - {e}")
```

<details>

<summary>
Inconsistent validation between integer and float inputs
</summary>
```
Integer -5: ValueError - 'test_param' must be an integer >=0
Float -5.0: -5
```
</details>

## Why This Is A Bug

This violates the explicit documentation contract of the `validate_integer` function. The function's docstring clearly states that the `min_val` parameter specifies the "Minimum allowed value (val < min_val will result in a ValueError)". This contract makes no distinction between integer and float inputs - both are listed as acceptable input types in the function signature (`val: int or float`).

The inconsistency occurs because the code has two separate validation paths:
1. For integer inputs: The code correctly checks `val >= min_val` in line 553
2. For float inputs: The code only verifies that the float can be safely cast to an integer (line 550-552) but never validates against `min_val`

This creates a situation where `-5` raises a ValueError while `-5.0` silently passes through and returns `-5`, despite both representing the same mathematical value. The error message itself states that the value "must be an integer >={min_val:d}", but this constraint is not enforced for the float code path.

## Relevant Context

The `validate_integer` function is an internal utility in pandas used to validate parameters in IO operations, particularly in CSV parsing. Common parameters validated by this function include:
- `nrows`: Number of rows to read from a file
- `chunksize`: Size of chunks when reading files iteratively
- `skipfooter`: Number of lines to skip at the bottom of the file

These parameters logically require non-negative integer values, making the `min_val` constraint critical for proper validation. While users are unlikely to intentionally pass negative floats like `-5.0` for these parameters, the inconsistency could arise from:
- Calculated values from formulas that produce float results
- Data passed from other systems or APIs that use floats
- User errors that should be caught consistently

The function is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/parsers/readers.py` at lines 527-556.

## Proposed Fix

```diff
--- a/pandas/io/parsers/readers.py
+++ b/pandas/io/parsers/readers.py
@@ -549,6 +549,8 @@ def validate_integer(
     if is_float(val):
         if int(val) != val:
             raise ValueError(msg)
         val = int(val)
+        if val < min_val:
+            raise ValueError(msg)
     elif not (is_integer(val) and val >= min_val):
         raise ValueError(msg)
```