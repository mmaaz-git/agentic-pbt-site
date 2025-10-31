# Bug Report: pandas.core.util.hashing.hash_array Signed Zero Hash Inequality

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`hash_array()` violates the fundamental hash invariant by producing different hash values for `0.0` and `-0.0`, despite these values being equal according to IEEE 754, numpy, and Python.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
@settings(max_examples=500)
def test_hash_equal_values_have_equal_hashes(values):
    arr = np.array(values)

    for i in range(len(arr)):
        if arr[i] == 0.0:
            arr_pos = arr.copy()
            arr_pos[i] = 0.0
            arr_neg = arr.copy()
            arr_neg[i] = -0.0

            hash_pos = hash_array(arr_pos)
            hash_neg = hash_array(arr_neg)

            assert np.array_equal(hash_pos, hash_neg), \
                f"Equal arrays should have equal hashes: {arr_pos} vs {arr_neg}"

if __name__ == "__main__":
    test_hash_equal_values_have_equal_hashes()
```

<details>

<summary>
**Failing input**: `values=[0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 24, in <module>
    test_hash_equal_values_have_equal_hashes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 6, in test_hash_equal_values_have_equal_hashes
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 20, in test_hash_equal_values_have_equal_hashes
    assert np.array_equal(hash_pos, hash_neg), \
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
AssertionError: Equal arrays should have equal hashes: [0.] vs [-0.]
Falsifying example: test_hash_equal_values_have_equal_hashes(
    values=[0.0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

# Test case demonstrating the bug
arr_pos = np.array([0.0])
arr_neg = np.array([-0.0])

print("=== Testing hash_array with signed zeros ===")
print()
print(f"arr_pos = np.array([0.0])")
print(f"arr_neg = np.array([-0.0])")
print()
print(f"Arrays equal (np.array_equal): {np.array_equal(arr_pos, arr_neg)}")
print(f"Element equal (0.0 == -0.0): {0.0 == -0.0}")
print()
print(f"hash_array([0.0]):  {hash_array(arr_pos)}")
print(f"hash_array([-0.0]): {hash_array(arr_neg)}")
print()
print(f"Hashes equal: {np.array_equal(hash_array(arr_pos), hash_array(arr_neg))}")
print()
print("This violates the hash invariant: if a == b, then hash(a) must equal hash(b)")
print()
print("For comparison, Python's built-in hash function handles this correctly:")
print(f"hash(0.0):  {hash(0.0)}")
print(f"hash(-0.0): {hash(-0.0)}")
print(f"hash(0.0) == hash(-0.0): {hash(0.0) == hash(-0.0)}")
```

<details>

<summary>
Hash values differ for equal arrays containing signed zeros
</summary>
```
=== Testing hash_array with signed zeros ===

arr_pos = np.array([0.0])
arr_neg = np.array([-0.0])

Arrays equal (np.array_equal): True
Element equal (0.0 == -0.0): True

hash_array([0.0]):  [0]
hash_array([-0.0]): [2720858781877447050]

Hashes equal: False

This violates the hash invariant: if a == b, then hash(a) must equal hash(b)

For comparison, Python's built-in hash function handles this correctly:
hash(0.0):  0
hash(-0.0): 0
hash(0.0) == hash(-0.0): True
```
</details>

## Why This Is A Bug

This bug violates the fundamental mathematical property of hash functions: **if two values are equal, their hashes must be equal**. This is a universal requirement across all programming languages and hash function implementations.

The IEEE 754 floating-point standard defines positive zero (`0.0`) and negative zero (`-0.0`) as distinct bit patterns that must compare as equal. Both numpy and Python correctly implement this:
- `0.0 == -0.0` returns `True`
- `np.array_equal([0.0], [-0.0])` returns `True`
- Python's `hash(0.0) == hash(-0.0)` returns `True`

However, pandas' `hash_array()` produces different hash values for these equal values, which causes incorrect behavior in:
- **Groupby operations**: Rows with `0.0` and `-0.0` in grouping columns would incorrectly be placed in different groups
- **Deduplication**: Values `0.0` and `-0.0` would not be deduplicated despite being equal
- **Hash-based joins**: Joins on columns containing signed zeros would miss matches
- **Index lookups**: Index positions with `0.0` wouldn't match lookups with `-0.0`

## Relevant Context

The bug occurs in the `_hash_ndarray` function at lines 305-306 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/util/hashing.py`:

```python
elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
    vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
```

This code directly interprets the bit representation of floating-point numbers as unsigned integers. Since `0.0` has all bits set to zero while `-0.0` has only the sign bit set, they produce different hash values:
- `0.0` bit pattern: `0x0000000000000000` → hash: 0
- `-0.0` bit pattern: `0x8000000000000000` → hash: 2720858781877447050

The hash_array function is a core utility used throughout pandas for:
- `hash_pandas_object()` which hashes Series, DataFrames, and Index objects
- Internal operations requiring deterministic hashing of array data
- Performance-critical operations that depend on hash consistency

## Proposed Fix

The issue can be fixed by normalizing signed zeros before hashing. This ensures `-0.0` is converted to `0.0` before the bit representation is used:

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -303,6 +303,9 @@ def _hash_ndarray(
     elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
         vals = vals.view("i8").astype("u8", copy=False)
     elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
+        # Normalize signed zeros to ensure 0.0 and -0.0 hash the same
+        if issubclass(dtype.type, np.floating):
+            vals = vals + 0.0  # Forces -0.0 to become +0.0
         vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
     else:
         # object dtypes handled below
```