# Bug Report: pandas.io.formats.format.format_percentiles Rounds Non-Zero Values to 0%

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_percentiles` function incorrectly rounds very small non-zero percentiles to "0%" without any decimal places, violating its documented contract that states "no entry is *rounded* to 0% or 100%" unless already equal to it, and that "Any non-integer is always rounded to at least 1 decimal place."

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.io.formats.format import format_percentiles


@given(
    percentile=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
def test_format_percentiles_non_integer_has_decimal(percentile):
    assume(percentile not in [0.0, 1.0])
    assume(not (percentile * 100).is_integer())

    result = format_percentiles([percentile])

    assert len(result) == 1
    value_str = result[0].rstrip('%')
    assert '.' in value_str, f"Non-integer {percentile} should have decimal: {result[0]}"


if __name__ == "__main__":
    test_format_percentiles_non_integer_has_decimal()
```

<details>

<summary>
**Failing input**: `4.827995913896854e-25`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 20, in <module>
    test_format_percentiles_non_integer_has_decimal()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 6, in test_format_percentiles_non_integer_has_decimal
    percentile=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 16, in test_format_percentiles_non_integer_has_decimal
    assert '.' in value_str, f"Non-integer {percentile} should have decimal: {result[0]}"
           ^^^^^^^^^^^^^^^^
AssertionError: Non-integer 4.827995913896854e-25 should have decimal: 0%
Falsifying example: test_format_percentiles_non_integer_has_decimal(
    percentile=4.827995913896854e-25,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py:1597

```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.format import format_percentiles

percentile = 1.401298464324817e-45
result = format_percentiles([percentile])

print(f"Input percentile: {percentile}")
print(f"Output: {result[0]}")
print(f"Is percentile zero: {percentile == 0.0}")
print(f"Is percentile * 100 an integer: {(percentile * 100).is_integer()}")
print()
print("According to docstring:")
print("1. 'no entry is *rounded* to 0% or 100%' (unless already equal to it)")
print("2. 'Any non-integer is always rounded to at least 1 decimal place'")
print()
print(f"Bug: Non-zero value {percentile} is formatted as '{result[0]}' with no decimal")
```

<details>

<summary>
Contract violation: Non-zero value formatted as "0%" without decimal
</summary>
```
Input percentile: 1.401298464324817e-45
Output: 0%
Is percentile zero: False
Is percentile * 100 an integer: False

According to docstring:
1. 'no entry is *rounded* to 0% or 100%' (unless already equal to it)
2. 'Any non-integer is always rounded to at least 1 decimal place'

Bug: Non-zero value 1.401298464324817e-45 is formatted as '0%' with no decimal

```
</details>

## Why This Is A Bug

The function's docstring explicitly documents two contracts that are being violated:

1. **"no entry is *rounded* to 0% or 100%"** (emphasis on *rounded*, unless already equal to it): The function rounds values like `1.401298464324817e-45` to "0%" even though `percentile == 0.0` evaluates to `False`. The value is mathematically non-zero but gets rounded to exactly "0%".

2. **"Any non-integer is always rounded to at least 1 decimal place"**: When `percentile * 100` is checked with `.is_integer()`, it returns `False`, indicating this is not an integer percentage. Yet the output is "0%" with no decimal places.

The bug affects all percentiles smaller than approximately 1e-10. Values from 1e-10 down to the smallest positive float are all incorrectly formatted as "0%" without any decimal places.

## Relevant Context

The issue stems from the `get_precision` function in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/formats/format.py` which calculates the precision needed for formatting. When dealing with extremely small values, the calculation `prec = -np.floor(np.log10(np.min(diff))).astype(int)` can overflow or produce very large precision values that cause the rounding logic to fail.

Testing shows the following behavior boundaries:
- Values ≤ 1e-10: formatted as "0%" (incorrect)
- Values from 1e-9 to 1e-7: formatted using scientific notation (e.g., "1e-07%")
- Values ≥ 1e-6: formatted with decimal places (e.g., "0.0001%")

This function is used internally by pandas' `describe()` method when displaying percentile statistics, potentially misleading users about the actual distribution of their data when dealing with very small values.

## Proposed Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1587,6 +1587,13 @@ def format_percentiles(
         raise ValueError("percentiles should all be in the interval [0,1]")

     percentiles = 100 * percentiles
+
+    # Handle extremely small non-zero percentiles to avoid rounding to exactly 0%
+    # Any value smaller than 1e-8 (0.000001%) but non-zero should show at least one decimal
+    MIN_THRESHOLD = 1e-8
+    if np.any((percentiles > 0) & (percentiles < MIN_THRESHOLD)):
+        # Use scientific notation or ensure at least one decimal place
+        return [f"{p:.1e}%" if p < MIN_THRESHOLD and p > 0 else f"{p}%" for p in percentiles]
+
     prec = get_precision(percentiles)
     percentiles_round_type = percentiles.round(prec).astype(int)
```