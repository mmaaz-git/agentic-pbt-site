# Bug Report: dask.dataframe.hyperloglog compute_first_bit

**Target**: `dask.dataframe.hyperloglog.compute_first_bit`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `compute_first_bit` function in HyperLogLog computes the position of the **rightmost** set bit + 1, but the HyperLogLog algorithm requires the number of **leading zeros** + 1. This fundamental error affects cardinality estimation accuracy.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np

def compute_first_bit(a):
    bits = np.bitwise_and.outer(a, 1 << np.arange(32))
    bits = bits.cumsum(axis=1).astype(bool)
    return 33 - bits.sum(axis=1)

@given(st.lists(st.integers(min_value=1, max_value=2**32-1), min_size=1, max_size=100))
def test_compute_first_bit_should_count_leading_zeros(values):
    a = np.array(values, dtype=np.uint32)
    actual = compute_first_bit(a)

    expected = np.array([33 - v.bit_length() for v in values])

    assert np.array_equal(actual, expected), f"Should count leading zeros, not trailing"
```

**Failing input**: `[1]`
- Actual: `1` (position of rightmost bit + 1)
- Expected: `32` (leading zeros + 1)

## Reproducing the Bug

```python
import numpy as np

def compute_first_bit(a):
    bits = np.bitwise_and.outer(a, 1 << np.arange(32))
    bits = bits.cumsum(axis=1).astype(bool)
    return 33 - bits.sum(axis=1)

value = 1
a = np.array([value], dtype=np.uint32)
result = compute_first_bit(a)[0]

print(f"Value: {value} (binary: {format(value, '032b')})")
print(f"compute_first_bit result: {result}")
print(f"Expected (leading zeros + 1): {32}")

assert result == 32, f"Expected 32, got {result}"
```

Output:
```
Value: 1 (binary: 00000000000000000000000000000001)
compute_first_bit result: 1
Expected (leading zeros + 1): 32
AssertionError: Expected 32, got 1
```

## Why This Is A Bug

The HyperLogLog algorithm (Flajolet et al. 2007) requires computing ρ(w), defined as the position of the leftmost 1-bit in the binary representation of w, which equals (leading zeros + 1).

For a hash value like `0x00000001`:
- **Leading zeros**: 31
- **ρ (required)**: 32
- **Rightmost bit position**: 0
- **Current result**: 1

The function computes the position of the **rightmost** (least significant) bit + 1, but HyperLogLog needs the number of **leading** (leftmost) zeros + 1.

### Static Analysis

For value = 1:
```python
bits = bitwise_and.outer([1], [1, 2, 4, ...])  # = [1, 0, 0, ...]
cumsum = [1, 1, 1, ...]  # 32 ones
as_bool = [True, True, ...]  # 32 True values
sum = 32
result = 33 - 32 = 1
```

This computes: `33 - (number of bits where cumsum > 0)`
= `33 - (number of bits at or above rightmost set bit)`
= `position of rightmost set bit + 1`

But HyperLogLog needs: `number of leading zeros + 1`

## Fix

```diff
--- a/dask/dataframe/hyperloglog.py
+++ b/dask/dataframe/hyperloglog.py
@@ -18,10 +18,15 @@ from pandas.util import hash_pandas_object


 def compute_first_bit(a):
-    "Compute the position of the first nonzero bit for each int in an array."
-    # TODO: consider making this less memory-hungry
-    bits = np.bitwise_and.outer(a, 1 << np.arange(32))
-    bits = bits.cumsum(axis=1).astype(bool)
-    return 33 - bits.sum(axis=1)
+    "Compute the position of the leftmost 1-bit (leading zeros + 1) for HyperLogLog."
+    result = np.empty(len(a), dtype=np.intp)
+    for i, val in enumerate(a):
+        if val == 0:
+            result[i] = 33
+        else:
+            # Count leading zeros: 32 - bit_length gives leading zeros
+            # Add 1 for HyperLogLog rho function
+            result[i] = 33 - val.bit_length()
+    return result
```

Alternatively, a more vectorized fix:

```diff
--- a/dask/dataframe/hyperloglog.py
+++ b/dask/dataframe/hyperloglog.py
@@ -18,10 +18,11 @@ from pandas.util import hash_pandas_object


 def compute_first_bit(a):
-    "Compute the position of the first nonzero bit for each int in an array."
-    # TODO: consider making this less memory-hungry
-    bits = np.bitwise_and.outer(a, 1 << np.arange(32))
-    bits = bits.cumsum(axis=1).astype(bool)
-    return 33 - bits.sum(axis=1)
+    "Compute the position of the leftmost 1-bit (leading zeros + 1) for HyperLogLog."
+    # For each value, compute 32 - bit_length() to get leading zeros, then add 1
+    result = np.zeros(len(a), dtype=np.intp)
+    nonzero = a != 0
+    result[nonzero] = 33 - np.floor(np.log2(a[nonzero])).astype(np.intp) - 1
+    result[~nonzero] = 33
+    return result
```