# Bug Report: pandas.qcut Crash with Subnormal Floats

**Target**: `pandas.core.reshape.tile.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.qcut` crashes with ValueError when the input data contains subnormal (denormalized) floating-point numbers mixed with normal values, instead of handling them correctly or raising a clear error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas as pd
import numpy as np

@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1000),
        min_size=20,
        max_size=200,
        unique=True
    ),
    q=st.integers(min_value=2, max_value=10)
)
def test_qcut_equal_sized_buckets(data, q):
    """qcut should produce approximately equal-sized buckets."""
    x = np.array(data)
    assume(len(x) >= q * 2)

    result = pd.qcut(x, q=q)
    value_counts = result.value_counts()

    expected_size = len(x) / len(value_counts)
    for count in value_counts:
        ratio = count / expected_size
        assert 0.5 <= ratio <= 2.0
```

**Failing input**: `data=[1e-310, 1.0, 2.0, 3.0], q=2` (subnormal float mixed with normal values)

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

data = [1e-310, 1.0, 2.0, 3.0]
result = pd.qcut(data, q=2)
```

**Output:**
```
ValueError: missing values must be missing in the same location both left and right sides
```

The bug occurs specifically when:
- Input contains subnormal floats (values < 2.23e-308, the smallest normal float)
- These are mixed with normal-magnitude values
- A warning "invalid value encountered in divide" precedes the crash

## Why This Is A Bug

Subnormal (denormalized) floats are valid IEEE 754 floating-point numbers that represent very small values between zero and the smallest normal float. They may appear in scientific computing applications dealing with extremely small probabilities, physical constants, or numerical precision issues.

`qcut` should either:
1. Handle subnormal floats correctly (preferred), or
2. Raise a clear, informative error message explaining the limitation

Instead, it crashes with a confusing error message about "missing values" when no NaN values are present in the input.

## Fix

The issue appears to be in the quantile calculation when values span an extremely large dynamic range. The division warning suggests numerical instability when computing breaks or precision formatting.

A potential fix would be to:
1. Add input validation to detect extreme dynamic ranges
2. Improve numerical stability in the quantile break calculation
3. Add a clear error message if subnormal floats cannot be handled

The root cause is likely in `/pandas/core/reshape/tile.py` in the `_bins_to_cuts` or `_format_labels` functions where the interval breaks are computed and formatted.
