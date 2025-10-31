# Bug Report: pandas.core.methods.describe.format_percentiles - Violates Uniqueness Property for Tiny Differences

**Target**: `pandas.core.methods.describe.format_percentiles`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates two documented properties when handling extremely small values (below approximately 1e-10):

1. **Uniqueness violation**: Different percentiles collapse to the same formatted string, violating the claim that "if any two elements of percentiles differ, they remain different after rounding"
2. **Rounding to 0%**: Non-zero percentiles get rounded to "0%", violating the claim that "no entry is *rounded* to 0% or 100%" unless exactly 0.0 or 1.0

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.methods.describe import format_percentiles

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=50))
def test_format_percentiles_different_inputs_remain_different(percentiles):
    """
    Property: If two percentiles differ, they should have different formatted strings
    """
    unique_percentiles = list(set(percentiles))

    if len(unique_percentiles) <= 1:
        return

    formatted = format_percentiles(percentiles)
    unique_formatted = set(formatted)

    if len(unique_percentiles) > 1:
        assert len(unique_formatted) > 1, (
            f"Different percentiles collapsed to same format: "
            f"input had {len(unique_percentiles)} unique values, "
            f"but output has only {len(unique_formatted)} unique strings: {unique_formatted}"
        )
```

**Failing input**: `percentiles=[0.0, 3.6340605919844266e-284]`

## Reproducing the Bug

```python
from pandas.core.methods.describe import format_percentiles

percentiles = [0.0, 3.6340605919844266e-284]
result = format_percentiles(percentiles)

print(f"Input: {percentiles}")
print(f"Output: {result}")

assert len(set(percentiles)) == 2
assert len(set(result)) == 2
```

Output:
```
Input: [0.0, 3.6340605919844266e-284]
Output: ['0%', '0%']
AssertionError: Expected 2 unique outputs, but got 1: {'0%'}
```

Second manifestation - non-zero rounding to 0%:
```python
from pandas.core.methods.describe import format_percentiles

result = format_percentiles([1.401298464324817e-45])
print(f"Result: {result}")
assert result[0] != '0%', f"Non-zero percentile rounded to 0%"
```

Output:
```
Result: ['0%']
AssertionError: Non-zero percentile rounded to 0%
```

## Why This Is A Bug

The function's docstring explicitly makes two promises that are violated:

**Promise 1**:
> Rounding precision is chosen so that: (1) if any two elements of ``percentiles`` differ, they remain different after rounding

**Violation**: The two distinct values `0.0` and `3.6e-284` both format to `'0%'`.

**Promise 2**:
> (2) no entry is *rounded* to 0% or 100%.

**Violation**: The non-zero value `1.401298464324817e-45` gets rounded to `'0%'`.

The root cause is in the `get_precision` helper function, which calculates precision based on the minimum difference between percentiles:

```python
prec = -np.floor(np.log10(np.min(diff))).astype(int)
```

For extremely small differences, this produces very large precision values that exceed Python's float formatting capabilities, causing the values to collapse to the same representation.

## Fix

The function should either:

1. **Fix the implementation** to handle tiny differences correctly by capping precision at a reasonable maximum
2. **Update the documentation** to clarify that the uniqueness property only holds for "reasonable" differences

Here's a possible fix for option 1:

```diff
diff --git a/pandas/io/formats/format.py b/pandas/io/formats/format.py
index xxxxx..yyyyy 100644
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -xxxx,7 +xxxx,8 @@ def get_precision(array: np.ndarray | Sequence[float]) -> int:
     diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
     diff = abs(diff)
     prec = -np.floor(np.log10(np.min(diff))).astype(int)
-    prec = max(1, prec)
+    # Cap precision at a reasonable maximum to avoid formatting issues
+    prec = max(1, min(prec, 15))
     return prec
```

Note: The exact maximum precision value (15 in this example) should be chosen based on what Python's float formatting can reliably handle.