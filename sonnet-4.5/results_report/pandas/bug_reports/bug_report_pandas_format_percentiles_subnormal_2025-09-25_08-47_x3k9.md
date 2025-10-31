# Bug Report: pandas.io.formats.format.format_percentiles Subnormal Float Handling

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_percentiles` function produces invalid output ("nan%" or "inf%") and RuntimeWarnings when given valid but very small percentile values (subnormal floats in the range [0, 1]).

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import warnings
import pytest
from pandas.io.formats.format import format_percentiles


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False), min_size=1, max_size=20, unique=True))
@settings(max_examples=1000)
def test_format_percentiles_no_warnings(percentiles):
    """
    Property: format_percentiles should not generate RuntimeWarnings for valid inputs.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        result = format_percentiles(percentiles)

        runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]

        if runtime_warnings:
            warning_messages = [str(warning.message) for warning in runtime_warnings]
            pytest.fail(
                f"format_percentiles generated RuntimeWarnings for percentiles {percentiles}:\n" +
                "\n".join(warning_messages)
            )
```

**Failing input**: `percentiles=[0.625, 2.225073858507203e-309]` (and other subnormal floats like 1e-320, 5e-324)

## Reproducing the Bug

```python
import warnings
from pandas.io.formats.format import format_percentiles
from pandas.util._validators import validate_percentile

percentile = 1e-320

validate_percentile(percentile)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)
    result = format_percentiles([percentile])

    print(f"Input: {percentile}")
    print(f"Output: {result}")
    print(f"Warnings: {[str(x.message) for x in w]}")
```

Output:
```
Input: 1e-320
Output: ['nan%']
Warnings: ['invalid value encountered in divide', 'invalid value encountered in cast', 'invalid value encountered in divide']
```

Real-world impact on `describe()`:
```python
import pandas as pd

series = pd.Series([1, 2, 3, 4, 5])
result = series.describe(percentiles=[2.225073858507203e-309])
print(result)
```

Output:
```
count    5.000000
mean     3.000000
std      1.581139
min      1.000000
0%       1.000000
inf%     3.000000
max      5.000000
dtype: float64
```

## Why This Is A Bug

1. **Valid inputs rejected**: The percentiles pass `validate_percentile()` - they are valid values in [0, 1]
2. **Invalid output**: The function produces "nan%" or "inf%" strings, which are not valid percentage representations
3. **RuntimeWarnings**: The function generates warnings about overflow and invalid values during computation
4. **Broken contract**: The docstring promises to format percentiles in [0, 1], but fails for subnormal floats
5. **Real impact**: When users call `.describe(percentiles=[very_small_value])`, they get "inf%" or "nan%" in the index, which breaks downstream code and user expectations

## Fix

The issue occurs when `percentiles * 100` is computed with subnormal floats. The multiplication can cause precision issues that propagate through subsequent operations. The function should handle subnormal floats more carefully.

Suggested fix: Add explicit handling for very small percentiles by clamping them or formatting them directly:

```diff
diff --git a/pandas/io/formats/format.py b/pandas/io/formats/format.py
index 1234567..abcdef0 100644
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1589,6 +1589,11 @@ def format_percentiles(

     percentiles = 100 * percentiles
     prec = get_precision(percentiles)
+
+    # Handle subnormal floats that may cause overflow/underflow
+    # Clamp very small values to 0 to avoid NaN/inf in output
+    percentiles = np.where(np.abs(percentiles) < 1e-300, 0.0, percentiles)
+
     percentiles_round_type = percentiles.round(prec).astype(int)

     int_idx = np.isclose(percentiles_round_type, percentiles)
```

Alternative: The function could check for subnormal floats before the multiplication and format them as "0%" directly, avoiding the problematic arithmetic.