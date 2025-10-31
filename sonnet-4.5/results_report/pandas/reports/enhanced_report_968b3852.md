# Bug Report: pandas.core.util.hashing.hash_array Ignores hash_key Parameter for Numeric Arrays

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_array` function accepts a `hash_key` parameter that should affect hash output according to its API signature, but this parameter is completely ignored when hashing numeric arrays (integers, floats, booleans, datetime, timedelta), violating the API contract.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.core.util.hashing import hash_array


@given(st.lists(st.integers(), min_size=1, max_size=50))
def test_hash_array_key_ignored_for_numeric_arrays(values):
    arr = np.array(values)
    hash_key1 = '0' * 16
    hash_key2 = '1' * 16

    hash1 = hash_array(arr, hash_key=hash_key1, categorize=False)
    hash2 = hash_array(arr, hash_key=hash_key2, categorize=False)

    assert not np.array_equal(hash1, hash2), \
        f"Different hash keys should produce different hashes for numeric arrays. Input: {values}"


if __name__ == "__main__":
    test_hash_array_key_ignored_for_numeric_arrays()
```

<details>

<summary>
**Failing input**: `[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 20, in <module>
    test_hash_array_key_ignored_for_numeric_arrays()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 7, in test_hash_array_key_ignored_for_numeric_arrays
    def test_hash_array_key_ignored_for_numeric_arrays(values):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 15, in test_hash_array_key_ignored_for_numeric_arrays
    assert not np.array_equal(hash1, hash2), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Different hash keys should produce different hashes for numeric arrays. Input: [0]
Falsifying example: test_hash_array_key_ignored_for_numeric_arrays(
    values=[0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/util/hashing.py:306
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

# Create a simple numeric array
arr = np.array([1, 2, 3])

# Hash with two different keys
hash1 = hash_array(arr, hash_key='0' * 16)
hash2 = hash_array(arr, hash_key='1' * 16)

print("Array:", arr)
print("Hash with key '0'*16:", hash1)
print("Hash with key '1'*16:", hash2)
print("Are they equal?:", np.array_equal(hash1, hash2))

# Additional test with different array types
print("\nTesting with different numeric types:")

# Integer array
int_arr = np.array([42])
int_hash1 = hash_array(int_arr, hash_key='0' * 16)
int_hash2 = hash_array(int_arr, hash_key='1' * 16)
print(f"Integer array {int_arr}: hashes equal? {np.array_equal(int_hash1, int_hash2)}")

# Float array
float_arr = np.array([3.14])
float_hash1 = hash_array(float_arr, hash_key='0' * 16)
float_hash2 = hash_array(float_arr, hash_key='1' * 16)
print(f"Float array {float_arr}: hashes equal? {np.array_equal(float_hash1, float_hash2)}")

# Boolean array
bool_arr = np.array([True, False])
bool_hash1 = hash_array(bool_arr, hash_key='0' * 16)
bool_hash2 = hash_array(bool_arr, hash_key='1' * 16)
print(f"Boolean array {bool_arr}: hashes equal? {np.array_equal(bool_hash1, bool_hash2)}")

# String array (object dtype) - this should work correctly
str_arr = np.array(['hello', 'world'], dtype=object)
str_hash1 = hash_array(str_arr, hash_key='0' * 16)
str_hash2 = hash_array(str_arr, hash_key='1' * 16)
print(f"String array {str_arr}: hashes equal? {np.array_equal(str_hash1, str_hash2)}")
```

<details>

<summary>
Numeric arrays produce identical hashes despite different keys, while string arrays correctly produce different hashes
</summary>
```
Array: [1 2 3]
Hash with key '0'*16: [ 6238072747940578789 15839785061582574730  2185194620014831856]
Hash with key '1'*16: [ 6238072747940578789 15839785061582574730  2185194620014831856]
Are they equal?: True

Testing with different numeric types:
Integer array [42]: hashes equal? True
Float array [3.14]: hashes equal? True
Boolean array [ True False]: hashes equal? True
String array ['hello' 'world']: hashes equal? False
```
</details>

## Why This Is A Bug

The `hash_array` function signature explicitly includes `hash_key` as a parameter with a default value and documentation:

```python
def hash_array(
    vals: ArrayLike,
    encoding: str = "utf8",
    hash_key: str = _default_hash_key,  # Parameter is part of public API
    categorize: bool = True,
) -> npt.NDArray[np.uint64]:
    """
    ...
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    ...
    """
```

The function accepts the `hash_key` parameter for all input types without error or warning, creating a reasonable expectation that it affects the output. However, examining the implementation reveals that numeric arrays completely bypass the hash_key:

In `_hash_ndarray` (lines 301-306 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/util/hashing.py`):
- Boolean arrays are converted to u8 without using hash_key
- Datetime64/timedelta64 arrays are viewed as i8 then u8 without using hash_key
- Numeric types ≤ 8 bytes are viewed as unsigned then u8 without using hash_key
- After conversion, bitwise operations are applied (lines 334-339) that never incorporate hash_key

The hash_key is only used in two scenarios:
1. When values have object dtype (lines 326-331) - calls `hash_object_array` with hash_key
2. When categorize=True and values are categorized (lines 311-323) - passes hash_key to categorical hashing

This violates the principle of least surprise and the API contract. Users who provide different hash_key values reasonably expect different hash outputs for consistency and potential security/determinism requirements. The current behavior silently ignores a documented parameter without warning, which can lead to subtle bugs in applications that depend on hash determinism across different keys.

## Relevant Context

The pandas hashing module is used throughout the library for operations like:
- DataFrame/Series deduplication and uniqueness checks
- Hash-based indexing and lookups
- Data integrity verification
- Reproducible sampling with hash-based seeds

The `hash_key` parameter appears designed to provide deterministic but configurable hashing, potentially for:
- Security applications requiring different hash spaces
- Testing scenarios needing predictable but distinct hashes
- Multi-tenant systems requiring isolated hash spaces

Documentation link: https://pandas.pydata.org/docs/reference/api/pandas.util.hash_array.html

Affected code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/util/hashing.py:301-339`

## Proposed Fix

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -336,6 +336,11 @@ def _hash_ndarray(
     vals ^= vals >> 27
     vals *= np.uint64(0x94D049BB133111EB)
     vals ^= vals >> 31
+
+    # Incorporate hash_key for numeric arrays to maintain API consistency
+    if hash_key != _default_hash_key:
+        key_hash = hash_object_array(np.array([hash_key], dtype=object), hash_key, encoding)[0]
+        vals ^= key_hash
+
     return vals
```