# Bug Report: pandas.io.formats.format.format_percentiles Violates Uniqueness Preservation Contract

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_percentiles` function fails to preserve uniqueness when formatting percentiles that differ by less than approximately 1e-5 (as percentages), directly violating its documented guarantee that "if any two elements of percentiles differ, they remain different after rounding."

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from pandas.io.formats.format import format_percentiles

@settings(max_examples=1000)
@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_format_percentiles_uniqueness_preservation(percentiles):
    unique_input = np.unique(percentiles)
    assume(len(unique_input) > 1)

    formatted = format_percentiles(percentiles)
    unique_input_formatted = format_percentiles(unique_input)
    unique_output = list(dict.fromkeys(unique_input_formatted))

    assert len(unique_output) == len(unique_input), \
        f"Unique percentiles should remain unique after formatting. " \
        f"Input had {len(unique_input)} unique values but output has {len(unique_output)}."

if __name__ == "__main__":
    test_format_percentiles_uniqueness_preservation()
```

<details>

<summary>
**Failing input**: `[0.0, 2.8856429512297976e-68]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 20, in <module>
    test_format_percentiles_uniqueness_preservation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 6, in test_format_percentiles_uniqueness_preservation
    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 15, in test_format_percentiles_uniqueness_preservation
    assert len(unique_output) == len(unique_input), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Unique percentiles should remain unique after formatting. Input had 2 unique values but output has 1.
Falsifying example: test_format_percentiles_uniqueness_preservation(
    percentiles=[0.0, 2.8856429512297976e-68],
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.format import format_percentiles

# Test case from bug report
percentiles = [0.01, 0.0100001]
result = format_percentiles(percentiles)

print(f"Input:  {percentiles}")
print(f"Output: {result}")
print(f"Unique inputs:  {len(set(percentiles))}")
print(f"Unique outputs: {len(set(result))}")
print()
print("Expected: 2 unique outputs (since we have 2 unique inputs)")
print("Actual: Only 1 unique output - both values formatted as '1%'")
print()
print("This violates the documented guarantee that:")
print("'if any two elements of percentiles differ, they remain different after rounding'")
```

<details>

<summary>
Two distinct percentiles incorrectly produce identical formatted output
</summary>
```
Input:  [0.01, 0.0100001]
Output: ['1%', '1%']
Unique inputs:  2
Unique outputs: 1

Expected: 2 unique outputs (since we have 2 unique inputs)
Actual: Only 1 unique output - both values formatted as '1%'

This violates the documented guarantee that:
'if any two elements of percentiles differ, they remain different after rounding'
```
</details>

## Why This Is A Bug

The function's docstring explicitly promises in lines 1562-1565 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/formats/format.py`:

> "Rounding precision is chosen so that: (1) if any two elements of ``percentiles`` differ, they remain different after rounding"

This is an unambiguous contract that the function must preserve uniqueness - any two different input percentiles should produce different formatted strings. The implementation violates this guarantee due to the use of `np.isclose()` on line 1594 with default tolerances (rtol=1e-05, atol=1e-08).

When percentiles are multiplied by 100 to convert to percentage format (line 1590), values that differ by less than approximately 1e-5 are incorrectly treated as "close" to the same integer value. For example, both 1.0 and 1.00001 (after multiplication) are considered close to integer 1, causing both to format as '1%' instead of preserving their distinction.

The function's example in the docstring even demonstrates preserving uniqueness for values 0.01999 and 0.02001, suggesting the developers intended to support this level of precision. The current implementation fails to meet this documented specification.

## Relevant Context

The `format_percentiles` function is an internal formatting utility in pandas, primarily used by methods like `DataFrame.describe()` to display percentile statistics. While it's not directly exposed in the public API, it's still part of pandas' contract with users who rely on accurate statistical reporting.

The root cause is on line 1594 of the format.py file:
```python
int_idx = np.isclose(percentiles_round_type, percentiles)
```

The `np.isclose` function documentation: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
Default parameters: rtol=1e-05, atol=1e-08

These default tolerances are too permissive for maintaining the uniqueness guarantee, especially after the percentiles are scaled by 100.

## Proposed Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1591,7 +1591,7 @@ def format_percentiles(
     percentiles = 100 * percentiles
     prec = get_precision(percentiles)
     percentiles_round_type = percentiles.round(prec).astype(int)
-    int_idx = np.isclose(percentiles_round_type, percentiles)
+    int_idx = np.isclose(percentiles_round_type, percentiles, rtol=0, atol=1e-10)

     if np.all(int_idx):
         out = percentiles_round_type.astype(str)
```