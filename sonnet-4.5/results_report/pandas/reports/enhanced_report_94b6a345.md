# Bug Report: pandas.cut Silently Returns NaN for Subnormal Float Ranges

**Target**: `pandas.cut`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pandas.cut` silently returns all NaN values when binning data containing subnormal floats (values with range < ~1e-300), and raises confusing ValueError for negative subnormal values instead of properly binning the data or providing a clear error message.

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
        assert binned.notna().sum() == len(s), f"Expected all {len(s)} values to be binned, but got {binned.notna().sum()} non-NaN values"
    except ValueError:
        pass  # Some cases might raise ValueError, which is acceptable


if __name__ == "__main__":
    # Run the test with the specific failing input
    print("Running Hypothesis test with failing input...")
    print("values=[0.0, 2.2250738585e-313], bins=2")
    print()

    values = [0.0, 2.2250738585e-313]
    bins = 2

    # Manually test the logic from the property test
    s = pd.Series(values)
    try:
        binned = pd.cut(s, bins=bins)
        assert binned.notna().sum() == len(s), f"Expected all {len(s)} values to be binned, but got {binned.notna().sum()} non-NaN values"
        print("Test PASSED (unexpectedly)")
    except AssertionError as e:
        print(f"Test FAILED with assertion error: {e}")
    except ValueError as e:
        print(f"Test raised ValueError (acceptable): {e}")
    except Exception as e:
        print(f"Test FAILED with unexpected error: {type(e).__name__}: {e}")
```

<details>

<summary>
**Failing input**: `values=[0.0, 2.2250738585e-313], bins=2`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Running Hypothesis test with failing input...
values=[0.0, 2.2250738585e-313], bins=2

Test FAILED with assertion error: Expected all 2 values to be binned, but got 0 non-NaN values
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

print("=== Testing pandas.cut with subnormal floats ===")
print()

# Test case 1: Subnormal positive values
print("Test 1: Positive subnormal values")
print("-" * 40)
values = [0.0, 2.225e-313]
s = pd.Series(values)
print(f"Input values: {values}")
print(f"Input Series:\n{s}")
print()

result = pd.cut(s, bins=2)
print(f"Result of pd.cut(s, bins=2):\n{result}")
print(f"Result as list: {result.tolist()}")
print(f"All values are NaN: {result.isna().all()}")
print()

# Test case 2: Negative subnormal values (this should crash)
print("Test 2: Negative subnormal values")
print("-" * 40)
try:
    values_neg = [0.0, -2.225e-313]
    s_neg = pd.Series(values_neg)
    print(f"Input values: {values_neg}")
    print(f"Input Series:\n{s_neg}")
    print()

    result_neg = pd.cut(s_neg, bins=2)
    print(f"Result of pd.cut(s_neg, bins=2):\n{result_neg}")
    print(f"Result as list: {result_neg.tolist()}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
print()

# Test case 3: Slightly larger values that work
print("Test 3: Values that work (range >= 1e-300)")
print("-" * 40)
values_ok = [0.0, 1e-300]
s_ok = pd.Series(values_ok)
print(f"Input values: {values_ok}")
print(f"Input Series:\n{s_ok}")
print()

result_ok = pd.cut(s_ok, bins=2)
print(f"Result of pd.cut(s_ok, bins=2):\n{result_ok}")
print(f"Result as list: {result_ok.tolist()}")
print(f"All values are NaN: {result_ok.isna().all()}")
```

<details>

<summary>
Silent failure with all NaN values and confusing error for negative subnormal values
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
=== Testing pandas.cut with subnormal floats ===

Test 1: Positive subnormal values
----------------------------------------
Input values: [0.0, 2.225e-313]
Input Series:
0     0.000000e+00
1    2.225000e-313
dtype: float64

Result of pd.cut(s, bins=2):
0    NaN
1    NaN
dtype: category
Categories (0, interval[float64, right]): []
Result as list: [nan, nan]
All values are NaN: True

Test 2: Negative subnormal values
----------------------------------------
Input values: [0.0, -2.225e-313]
Input Series:
0     0.000000e+00
1   -2.225000e-313
dtype: float64

ERROR: ValueError: missing values must be missing in the same location both left and right sides

Test 3: Values that work (range >= 1e-300)
----------------------------------------
Input values: [0.0, 1e-300]
Input Series:
0     0.000000e+00
1    1.000000e-300
dtype: float64

Result of pd.cut(s_ok, bins=2):
0    (-1.0000000000000001e-303, 5e-301]
1      (5e-301, 9.999999999999999e-301]
dtype: category
Categories (2, interval[float64, right]): [(-1.0000000000000001e-303, 5e-301] < (5e-301, 9.999999999999999e-301]]
Result as list: [Interval(-1.0000000000000001e-303, 5e-301, closed='right'), Interval(5e-301, 9.999999999999999e-301, closed='right')]
All values are NaN: False
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Silent Data Loss**: The function silently converts all valid floating-point inputs to NaN without any warning or error message. The pandas.cut documentation states that "Any NA values will be NA in the result" but does not mention that valid non-NA inputs can become NA due to numerical precision issues.

2. **Inconsistent Error Handling**: For negative subnormal values, the function raises a confusing ValueError ("missing values must be missing in the same location both left and right sides") that has no apparent connection to the actual input issue. This error message is not documented and provides no guidance to users.

3. **Undocumented Limitation**: The documentation makes no mention of numerical precision limitations or minimum range requirements. Users reasonably expect the function to either successfully bin any valid floating-point data or raise a clear, informative error.

4. **Runtime Warnings Ignored**: NumPy raises "RuntimeWarning: invalid value encountered in divide" during execution, indicating the function is performing invalid mathematical operations, but pandas.cut proceeds anyway and returns invalid results.

5. **Threshold Behavior**: The function works correctly for ranges >= 1e-300 but fails silently for smaller ranges, creating a hidden threshold that users cannot discover without extensive testing.

## Relevant Context

- **Pandas Version**: 2.3.2
- **Python Version**: 3.13
- **Subnormal Floats**: IEEE 754 floating-point numbers smaller than the smallest normal number (~2.225e-308 for float64)
- **Use Cases**: Scientific computing applications dealing with quantum mechanics, particle physics, or other domains with extremely small measurements
- **Workaround**: Users can scale their data to a normal range before binning, then map the results back

The issue occurs during the bin edge calculation when the range (`max - min`) is so small that arithmetic operations produce NaN or infinity due to underflow. The function extends the range by 0.1% on each side (as documented), but this extension mechanism fails catastrophically with subnormal values.

Related pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.cut.html

## Proposed Fix

The bug requires adding validation for numerical stability before attempting to create bins. Here's a high-level fix approach:

1. Detect when the value range is in the subnormal region
2. Either raise an informative error or handle the case specially
3. Ensure consistent behavior for both positive and negative subnormal values

A simple detection and error approach would be to add validation in the cut function before bin calculation:

```diff
# In pandas/core/reshape/tile.py or equivalent location
def cut(...):
    # ... existing code ...

    # After determining x_min and x_max
    value_range = x_max - x_min

+   # Check for subnormal range that would cause numerical instability
+   if value_range > 0 and value_range < 1e-300:
+       raise ValueError(
+           f"Cannot bin values: range ({value_range:.2e}) is too small for numerical stability. "
+           f"Consider scaling your data before binning. "
+           f"pandas.cut requires a range >= 1e-300 to avoid floating-point underflow."
+       )

    # ... continue with existing binning logic ...
```

Alternatively, implement automatic scaling for subnormal ranges:

```diff
# More robust fix with automatic scaling
def cut(...):
    # ... existing code ...

    value_range = x_max - x_min

+   # Handle subnormal ranges by temporary scaling
+   scale_factor = 1.0
+   if value_range > 0 and value_range < 1e-300:
+       # Scale up to normal range
+       scale_factor = 1e-250 / value_range
+       x = x * scale_factor
+       if bins is not None and not np.iterable(bins):
+           # Recalculate min/max after scaling
+           x_min = np.nanmin(x)
+           x_max = np.nanmax(x)

    # ... perform binning on scaled values ...

+   # Scale intervals back if needed
+   if scale_factor != 1.0:
+       # Adjust the interval boundaries back to original scale
+       # (implementation details would need careful handling of interval objects)
```