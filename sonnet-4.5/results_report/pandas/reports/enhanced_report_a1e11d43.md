# Bug Report: pandas.io.formats.format.format_percentiles Violates Decimal Place Contract

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates its documented contract that "Any non-integer is always rounded to at least 1 decimal place". Non-integer percentiles that are close to integers after rounding are incorrectly formatted without decimal places.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.formats.format import format_percentiles

@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_format_percentiles_non_integer_has_decimal(percentile):
    """
    Property from docstring: "Any non-integer is always rounded to at least 1 decimal place"
    """
    formatted = format_percentiles([percentile])
    result = formatted[0]

    percent_value = percentile * 100
    is_integer_percent = abs(percent_value - round(percent_value)) < 1e-10

    if not is_integer_percent:
        assert '.' in result or ',' in result, \
            f"Non-integer percentile {percentile} formatted without decimal: {result}"

if __name__ == "__main__":
    # Run the test
    test_format_percentiles_non_integer_has_decimal()
```

<details>

<summary>
**Failing input**: `1e-10`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 22, in <module>
    test_format_percentiles_non_integer_has_decimal()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 5, in test_format_percentiles_non_integer_has_decimal
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 17, in test_format_percentiles_non_integer_has_decimal
    assert '.' in result or ',' in result, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Non-integer percentile 1e-10 formatted without decimal: 0%
Falsifying example: test_format_percentiles_non_integer_has_decimal(
    percentile=1e-10,
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.format import format_percentiles

# Test case that should demonstrate the bug
percentile = 0.8899967487632947
formatted = format_percentiles([percentile])

print(f"Input percentile: {percentile}")
print(f"As percentage: {percentile * 100}%")
print(f"Formatted output: {formatted[0]}")
print()

# Check if this is an integer percentage
percent_value = percentile * 100
is_integer_percent = abs(percent_value - round(percent_value)) < 1e-10
print(f"Is {percent_value}% an integer percentage? {is_integer_percent}")
print()

# The docstring states: "Any non-integer is always rounded to at least 1 decimal place"
# Since 88.99967487632946% is not an integer, it should have a decimal place
print("Expected: Output should contain a decimal point (e.g., '89.0%')")
print(f"Actual: {formatted[0]}")
print(f"Contains decimal? {'.' in formatted[0]}")
print()

# This should fail if the bug exists
try:
    assert '.' in formatted[0], f"Non-integer percentile {percentile} ({percent_value}%) formatted without decimal: {formatted[0]}"
    print("PASS: Decimal point found as expected")
except AssertionError as e:
    print(f"FAIL: {e}")
```

<details>

<summary>
AssertionError: Non-integer percentile formatted without decimal
</summary>
```
Input percentile: 0.8899967487632947
As percentage: 88.99967487632946%
Formatted output: 89%

Is 88.99967487632946% an integer percentage? False

Expected: Output should contain a decimal point (e.g., '89.0%')
Actual: 89%
Contains decimal? False

FAIL: Non-integer percentile 0.8899967487632947 (88.99967487632946%) formatted without decimal: 89%
```
</details>

## Why This Is A Bug

The function's docstring at line 1565 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/formats/format.py` explicitly states: "Any non-integer is always rounded to at least 1 decimal place."

The bug occurs because the function uses `np.isclose()` at line 1594 to determine which percentiles should be formatted as integers:

```python
int_idx = np.isclose(percentiles_round_type, percentiles)
```

This approach incorrectly classifies non-integer percentiles as integers when they round to values very close to integers. For example:
- Input: `0.8899967487632947` (88.99967487632946%)
- After rounding at certain precision: becomes close to 89
- `np.isclose()` returns True, treating it as an integer
- Output: "89%" instead of "89.0%" or a more precise representation

The same issue occurs with very small values like `1e-10` (0.00000001%) which gets formatted as "0%" instead of "0.0%".

## Relevant Context

The function serves to format percentiles for display in pandas DataFrame descriptions and quantile operations. The docstring provides clear examples showing that integer percentages (like 50%) should appear without decimals, while non-integers should always have at least one decimal place.

The function documentation can be found at:
- File: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/formats/format.py`
- Lines: 1546-1606

The bug affects precision in statistical reporting where the difference between "89%" and "89.0%" might indicate different levels of measurement precision or data quality.

## Proposed Fix

The issue can be resolved by using a stricter check to determine true integer percentages rather than using `np.isclose()`:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1591,7 +1591,8 @@ def format_percentiles(
     prec = get_precision(percentiles)
     percentiles_round_type = percentiles.round(prec).astype(int)

-    int_idx = np.isclose(percentiles_round_type, percentiles)
+    # Check if percentile is truly an integer (e.g., 0.5 -> 50%), not just close after rounding
+    int_idx = np.abs(percentiles - np.round(percentiles)) < 1e-10

     if np.all(int_idx):
         out = percentiles_round_type.astype(str)
```