# Bug Report: pandas.io.formats.format.format_percentiles RuntimeWarnings with Duplicate Percentile Values

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_percentiles` function produces RuntimeWarnings (divide by zero, invalid value encountered, overflow) when processing duplicate percentile values, despite correctly formatting the output and explicitly documenting that "Duplicates are allowed".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.formats.format as fmt
import warnings
import pytest


@given(
    value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    count=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=500)
def test_format_percentiles_no_warnings_for_duplicates(value, count):
    """When all percentiles are the same, the function should not produce warnings."""
    percentiles = [value] * count

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fmt.format_percentiles(percentiles)

        runtime_warnings = [warning for warning in w
                          if issubclass(warning.category, RuntimeWarning)]

        assert len(runtime_warnings) == 0, \
            f"Should not produce warnings for duplicate values: {[str(w.message) for w in runtime_warnings]}"

if __name__ == "__main__":
    test_format_percentiles_no_warnings_for_duplicates()
```

<details>

<summary>
**Failing input**: `value=0.0, count=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 27, in <module>
    test_format_percentiles_no_warnings_for_duplicates()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 8, in test_format_percentiles_no_warnings_for_duplicates
    value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 23, in test_format_percentiles_no_warnings_for_duplicates
    assert len(runtime_warnings) == 0, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Should not produce warnings for duplicate values: ['divide by zero encountered in log10', 'invalid value encountered in cast', 'overflow encountered in scalar negative']
Falsifying example: test_format_percentiles_no_warnings_for_duplicates(
    # The test always failed when commented parts were varied together.
    value=0.0,  # or any other generated value
    count=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas.io.formats.format as fmt
import warnings

warnings.simplefilter("always")
result = fmt.format_percentiles([0.5, 0.5, 0.5])
print(f"Result: {result}")
```

<details>

<summary>
RuntimeWarnings triggered despite correct output
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1614: RuntimeWarning: divide by zero encountered in log10
  prec = -np.floor(np.log10(np.min(diff))).astype(int)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1614: RuntimeWarning: invalid value encountered in cast
  prec = -np.floor(np.log10(np.min(diff))).astype(int)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1614: RuntimeWarning: overflow encountered in scalar negative
  prec = -np.floor(np.log10(np.min(diff))).astype(int)
Result: ['50%', '50%', '50%']
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Documentation contradiction**: The function's docstring explicitly states "Duplicates are allowed" (line 1575) and even provides an example with duplicates: `format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])`. The function should handle this documented use case without warnings.

2. **Unnecessary warnings for valid input**: The function accepts duplicate percentiles as valid input (no ValueError raised), produces correct output, but generates misleading RuntimeWarnings that suggest mathematical errors.

3. **Root cause**: The `get_precision` helper function (line 1609-1616) calculates differences between consecutive percentiles using `np.ediff1d`. When all values are identical, all differences are zero. It then computes `np.log10(0)` which mathematically produces negative infinity, triggering the cascade of warnings:
   - `divide by zero encountered in log10` when computing log10(0)
   - `invalid value encountered in cast` when trying to cast -inf to int
   - `overflow encountered in scalar negative` when negating the result

4. **Performance impact**: The function performs unnecessary expensive computations (log10, floor, type conversions) that generate warnings for what should be a trivial case (all values identical).

## Relevant Context

- The function is part of pandas' internal formatting utilities located at `/pandas/io/formats/format.py`
- Primary usage is through `pandas.core.methods.describe` module, though `DataFrame.describe()` itself prevents duplicate percentiles
- The function correctly handles the duplicate case in terms of output but inefficiently processes it through the general precision calculation path
- All duplicate values (whether 0.0, 0.25, 0.5, or 1.0) trigger the same three RuntimeWarnings
- The warnings occur specifically at line 1614 in the `get_precision` function when calculating precision for identical values

Documentation reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.format.format_percentiles.html

## Proposed Fix

Add a check for duplicate/identical values before calling `get_precision` to avoid the log10(0) computation entirely:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1587,6 +1587,11 @@ def format_percentiles(
     ):
         raise ValueError("percentiles should all be in the interval [0,1]")

+    # Handle the case where all percentiles are identical efficiently
+    unique_pcts = np.unique(percentiles)
+    if len(unique_pcts) == 1:
+        return [f"{int(percentiles[0] * 100)}%" for _ in percentiles]
+
     percentiles = 100 * percentiles
     prec = get_precision(percentiles)
     percentiles_round_type = percentiles.round(prec).astype(int)
```