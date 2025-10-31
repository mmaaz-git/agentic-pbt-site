# Bug Report: pandas.cut Subnormal Float Handling

**Target**: `pandas.cut`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.cut` silently returns all NaN values when binning data with a range in the subnormal float region (< ~1e-300), and can crash with ValueError for negative subnormal values.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import pandas as pd


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6), min_size=2, max_size=50),
    st.integers(min_value=2, max_value=10),
)
@settings(max_examples=500)
def test_cut_preserves_count(values, bins):
    assume(len(set(values)) >= 2)
    s = pd.Series(values)

    try:
        binned = pd.cut(s, bins=bins)
        assert binned.notna().sum() == len(s)
    except ValueError:
        pass
```

**Failing input**: `values=[0.0, 2.2250738585e-313], bins=2`

## Reproducing the Bug

```python
import pandas as pd

values = [0.0, 2.225e-313]
s = pd.Series(values)

result = pd.cut(s, bins=2)
print(result.tolist())

assert result.isna().all()
```

Output:
```
[nan, nan]
```

Additional failure modes:
```python
result = pd.cut(pd.Series([0.0, -2.225e-313]), bins=2)
```

Raises: `ValueError: missing values must be missing in the same location both left and right sides`

## Why This Is A Bug

The `pd.cut` function is designed to bin continuous data into discrete intervals. It should either:
1. Successfully bin any valid floating-point inputs, or
2. Raise a clear error if inputs are outside the supported range

Instead, it silently returns all NaN for values with ranges below ~1e-300, which could lead to:
- Silent data loss
- Misleading analysis results
- Difficult-to-debug issues in scientific computing pipelines

The issue affects values in the subnormal float range where `max(values) - min(values) < ~1e-300`. While rare in typical applications, users working with very small scientific measurements (particle physics, quantum mechanics, etc.) could encounter this.

## Fix

The root cause appears to be numeric precision issues in the binning calculation. When the range is very small, internal division operations produce invalid values (NaN/inf).

Suggested fixes:
1. **Detection and error**: Add a check for subnormal ranges and raise a clear ValueError
2. **Normalization**: Temporarily scale values to a normal range, perform binning, then map back
3. **Documentation**: At minimum, document this limitation

Example detection code to add:
```python
value_range = np.ptp(x)
if 0 < value_range < 1e-300:
    raise ValueError(
        f"Cannot bin values: range ({value_range}) is too small. "
        "pd.cut requires a range >= 1e-300 for numeric stability."
    )
```