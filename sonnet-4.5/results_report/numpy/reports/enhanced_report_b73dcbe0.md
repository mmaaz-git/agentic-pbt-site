# Bug Report: pandas format_percentiles produces invalid 'nan%' output with subnormal floats

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_percentiles()` function produces invalid 'nan%' strings instead of properly formatted percentages when given valid input that includes extremely small denormalized floating-point numbers (subnormal numbers like 5e-324).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.formats.format as fmt

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1))
@settings(max_examples=1000)
def test_format_percentiles_all_end_with_percent(percentiles):
    """
    Property: all formatted strings should end with '%'
    and not contain 'nan' or 'inf'
    """
    formatted = fmt.format_percentiles(percentiles)
    for f in formatted:
        assert f.endswith('%'), f"Formatted value '{f}' does not end with '%'"
        assert 'nan' not in f.lower(), f"Formatted value '{f}' contains 'nan'"

if __name__ == "__main__":
    test_format_percentiles_all_end_with_percent()
```

<details>

<summary>
**Failing input**: `[5e-324]`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1614: RuntimeWarning: divide by zero encountered in log10
  prec = -np.floor(np.log10(np.min(diff))).astype(int)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1614: RuntimeWarning: invalid value encountered in cast
  prec = -np.floor(np.log10(np.min(diff))).astype(int)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1614: RuntimeWarning: overflow encountered in scalar negative
  prec = -np.floor(np.log10(np.min(diff))).astype(int)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1592: RuntimeWarning: invalid value encountered in divide
  percentiles_round_type = percentiles.round(prec).astype(int)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1592: RuntimeWarning: invalid value encountered in cast
  percentiles_round_type = percentiles.round(prec).astype(int)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1605: RuntimeWarning: invalid value encountered in divide
  out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 17, in <module>
    test_format_percentiles_all_end_with_percent()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 5, in test_format_percentiles_all_end_with_percent
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 14, in test_format_percentiles_all_end_with_percent
    assert 'nan' not in f.lower(), f"Formatted value '{f}' contains 'nan'"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Formatted value 'nan%' contains 'nan'
Falsifying example: test_format_percentiles_all_end_with_percent(
    percentiles=[5e-324],
)
```
</details>

## Reproducing the Bug

```python
import pandas.io.formats.format as fmt

result = fmt.format_percentiles([0.625, 5e-324])
print(result)
```

<details>

<summary>
Output shows 'nan%' instead of valid percentage
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1605: RuntimeWarning: invalid value encountered in divide
  out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
['nan%', '0%']
```
</details>

## Why This Is A Bug

The function violates its documented contract in multiple ways:

1. **Invalid Output Format**: The docstring promises "formatted : list of strings" that are percentage values, but 'nan%' is not a valid formatted percentage string.

2. **Input Validation Passes**: The function explicitly validates that inputs are in [0,1], and 5e-324 passes this validation check. The function accepts the input as valid but then produces invalid output.

3. **Numeric Overflow Chain**: The bug occurs due to numeric overflow in the `get_precision()` helper function at line 1614. When processing 5e-324 (the smallest positive float64 value), the calculation `np.log10(np.min(diff))` produces extremely large negative values (around -321), which when negated and cast to int causes overflow. This cascades through the rounding operations, ultimately producing NaN values that get formatted as 'nan%'.

4. **Multiple Runtime Warnings**: The function generates several runtime warnings about divide by zero, invalid values in cast, and overflow - indicating the computation is failing internally while still returning a result.

5. **Downstream Impact**: Any code using `pandas.DataFrame.describe()` with data containing subnormal numbers would display 'nan%' to users, breaking reporting and analysis workflows.

## Relevant Context

- **5e-324** is the smallest positive normalized float64 value (approximately 4.94065645841247e-324)
- This is a denormalized/subnormal floating-point number per IEEE 754 standard
- The function is used internally by pandas for `describe()` output and quantile displays
- The warnings indicate the computation path: log10(tiny_value) → large negative → overflow on negation → NaN → 'nan%'
- Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py`

## Proposed Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1611,7 +1611,14 @@ def get_precision(array: np.ndarray | Sequence[float]) -> int:
     to_end = 100 - array[-1] if array[-1] < 100 else None
     diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
     diff = abs(diff)
-    prec = -np.floor(np.log10(np.min(diff))).astype(int)
+    min_diff = np.min(diff)
+    if min_diff == 0 or min_diff < 1e-15:
+        prec = 15  # Maximum reasonable precision for float display
+    else:
+        log_val = np.log10(min_diff)
+        if log_val < -15:
+            prec = 15
+        else:
+            prec = -np.floor(log_val).astype(int)
     prec = max(1, prec)
     return prec
```