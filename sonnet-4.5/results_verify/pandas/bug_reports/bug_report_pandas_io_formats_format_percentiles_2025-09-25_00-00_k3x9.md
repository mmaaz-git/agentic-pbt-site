# Bug Report: format_percentiles Violates Uniqueness and Boundary Guarantees

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates two properties explicitly stated in its docstring when given extremely small percentile values: (1) it fails to preserve uniqueness of distinct inputs, and (2) it rounds non-zero values to 0%.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.io.formats.format import format_percentiles
import numpy as np


@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20))
def test_format_percentiles_preserves_uniqueness(percentiles):
    percentiles_array = np.array(percentiles)
    unique_inputs = np.unique(percentiles_array)
    assume(len(unique_inputs) >= 2)

    formatted = format_percentiles(percentiles_array)

    input_to_output = {}
    for inp, outp in zip(percentiles_array, formatted):
        if inp in input_to_output:
            continue
        input_to_output[inp] = outp

    unique_outputs = set(input_to_output.values())
    assert len(unique_outputs) == len(unique_inputs)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True))
def test_format_percentiles_no_rounding_to_boundaries(percentile):
    assume(percentile > 0 and percentile < 1)
    formatted = format_percentiles([percentile])
    assert formatted[0] not in ['0%', '100%']
```

**Failing inputs**:
- Uniqueness: `[0.0, 2.2250738585072014e-308]`
- Boundary: `1.1754943508222875e-38`

## Reproducing the Bug

**Bug 1: Uniqueness violation**

```python
from pandas.io.formats.format import format_percentiles
import numpy as np

percentiles = np.array([0.0, 2.2250738585072014e-308])
formatted = format_percentiles(percentiles)

print(f"Input: {percentiles}")
print(f"Output: {formatted}")
print(f"Unique inputs: {len(np.unique(percentiles))}")
print(f"Unique outputs: {len(set(formatted))}")
```

Output:
```
Input: [0.00000000e+000 2.22507386e-308]
Output: ['0%', '0%']
Unique inputs: 2
Unique outputs: 1
```

**Bug 2: Boundary rounding violation**

```python
from pandas.io.formats.format import format_percentiles

percentile = 1.1754943508222875e-38
formatted = format_percentiles([percentile])

print(f"Input: {percentile} (non-zero: {percentile > 0})")
print(f"Output: {formatted}")
```

Output:
```
Input: 1.1754943508222875e-38 (non-zero: True)
Output: ['0%']
```

## Why This Is A Bug

The docstring explicitly states two properties that are violated:

1. **Uniqueness**: "Rounding precision is chosen so that: (1) if any two elements of ``percentiles`` differ, they remain different after rounding"
   - This is violated when `[0.0, 2.225e-308]` produces `['0%', '0%']`

2. **No boundary rounding**: "(2) no entry is *rounded* to 0% or 100%."
   - This is violated when non-zero value `1.175e-38` produces `'0%'`

The root cause is numerical instability in the `get_precision` helper function, which uses `np.log10(np.min(diff))`. When differences are extremely small:
- `log10` of tiny numbers produces very large negative values
- This causes overflow when negated: `-np.floor(np.log10(...))` overflows
- The overflowed precision value causes incorrect rounding behavior

## Fix

The issue can be fixed by adding bounds checking and special handling for extremely small differences in `get_precision`:

```diff
def get_precision(array: np.ndarray | Sequence[float]) -> int:
    to_begin = array[0] if array[0] > 0 else None
    to_end = 100 - array[-1] if array[-1] < 100 else None
    diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
    diff = abs(diff)
-   prec = -np.floor(np.log10(np.min(diff))).astype(int)
-   prec = max(1, prec)
+   min_diff = np.min(diff)
+   if min_diff == 0:
+       return 1
+   log_val = np.log10(min_diff)
+   if not np.isfinite(log_val):
+       return 15
+   prec = -np.floor(log_val).astype(int)
+   prec = np.clip(prec, 1, 15)
    return prec
```

Alternatively, the function could detect and reject percentiles that are too close together (within machine epsilon) to format distinctly, raising a more informative error.