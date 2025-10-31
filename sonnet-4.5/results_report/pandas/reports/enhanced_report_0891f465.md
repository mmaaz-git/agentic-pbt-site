# Bug Report: pandas.core.util.hashing.hash_array Silently Ignores hash_key Parameter for Numeric Arrays

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_array` function accepts a `hash_key` parameter but silently ignores it for numeric arrays (int, float, bool, datetime, timedelta), only applying it to object arrays, violating the API contract without warning or documentation.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings


@given(st.text(min_size=16, max_size=16))
@settings(max_examples=500)
def test_hash_array_different_hash_keys(hash_key):
    from pandas.core.util.hashing import hash_array

    arr = np.array([1, 2, 3])
    default_hash = hash_array(arr, hash_key="0123456789123456")
    custom_hash = hash_array(arr, hash_key=hash_key)

    if hash_key == "0123456789123456":
        assert np.array_equal(default_hash, custom_hash)
    else:
        assert not np.array_equal(default_hash, custom_hash)


if __name__ == "__main__":
    test_hash_array_different_hash_keys()
```

<details>

<summary>
**Failing input**: `hash_key='0000000000000000'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 21, in <module>
    test_hash_array_different_hash_keys()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 6, in test_hash_array_different_hash_keys
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 17, in test_hash_array_different_hash_keys
    assert not np.array_equal(default_hash, custom_hash)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_hash_array_different_hash_keys(
    hash_key='0000000000000000',
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

# Test with numeric arrays (int64)
arr = np.array([1, 2, 3, 4, 5])

hash1 = hash_array(arr, hash_key="0123456789123456")
hash2 = hash_array(arr, hash_key="AAAAAAAAAAAAAAAA")
hash3 = hash_array(arr, hash_key="different_key123")

print("Testing numeric arrays (int64):")
print(f"hash_key='0123456789123456': {hash1}")
print(f"hash_key='AAAAAAAAAAAAAAAA': {hash2}")
print(f"hash_key='different_key123': {hash3}")
print(f"All hashes identical: {np.array_equal(hash1, hash2) and np.array_equal(hash2, hash3)}")

# Test with object arrays
print("\nTesting object arrays:")
obj_arr = np.array(['a', 'b', 'c'], dtype=object)
obj_hash1 = hash_array(obj_arr, hash_key="0123456789123456", categorize=False)
obj_hash2 = hash_array(obj_arr, hash_key="AAAAAAAAAAAAAAAA", categorize=False)
print(f"hash_key='0123456789123456': {obj_hash1}")
print(f"hash_key='AAAAAAAAAAAAAAAA': {obj_hash2}")
print(f"Object arrays respect hash_key: {not np.array_equal(obj_hash1, obj_hash2)}")

# Test with other numeric types
print("\nTesting float arrays:")
float_arr = np.array([1.0, 2.0, 3.0])
float_hash1 = hash_array(float_arr, hash_key="0123456789123456")
float_hash2 = hash_array(float_arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"Float arrays respect hash_key: {not np.array_equal(float_hash1, float_hash2)}")

print("\nTesting bool arrays:")
bool_arr = np.array([True, False, True])
bool_hash1 = hash_array(bool_arr, hash_key="0123456789123456")
bool_hash2 = hash_array(bool_arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"Bool arrays respect hash_key: {not np.array_equal(bool_hash1, bool_hash2)}")

print("\nTesting datetime arrays:")
datetime_arr = np.array(['2021-01-01', '2021-01-02'], dtype='datetime64')
datetime_hash1 = hash_array(datetime_arr, hash_key="0123456789123456")
datetime_hash2 = hash_array(datetime_arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"Datetime arrays respect hash_key: {not np.array_equal(datetime_hash1, datetime_hash2)}")
```

<details>

<summary>
All numeric arrays produce identical hashes regardless of hash_key
</summary>
```
Testing numeric arrays (int64):
hash_key='0123456789123456': [ 6238072747940578789 15839785061582574730  2185194620014831856
 13232826040865663252 13168350753275463132]
hash_key='AAAAAAAAAAAAAAAA': [ 6238072747940578789 15839785061582574730  2185194620014831856
 13232826040865663252 13168350753275463132]
hash_key='different_key123': [ 6238072747940578789 15839785061582574730  2185194620014831856
 13232826040865663252 13168350753275463132]
All hashes identical: True

Testing object arrays:
hash_key='0123456789123456': [13950350942979735504 12688059582079114975 12544043241617149648]
hash_key='AAAAAAAAAAAAAAAA': [10461819242433495536  4426277128373134413  5515184766638266621]
Object arrays respect hash_key: True

Testing float arrays:
Float arrays respect hash_key: False

Testing bool arrays:
Bool arrays respect hash_key: False

Testing datetime arrays:
Datetime arrays respect hash_key: False
```
</details>

## Why This Is A Bug

The `hash_array` function violates its API contract in several critical ways:

1. **Silent Parameter Ignoring**: The function accepts `hash_key` for all input types but only uses it for object arrays. For numeric types (int, float, bool, datetime, timedelta), the parameter is completely ignored without warning. This violates the principle of least surprise - when a function accepts a parameter, users expect it to have an effect.

2. **Inconsistent Behavior Across Data Types**: The same parameter works differently based on the input dtype:
   - `hash_array(np.array([1, 2, 3]), hash_key="custom")` - hash_key silently ignored
   - `hash_array(np.array(['a','b','c'], dtype=object), hash_key="custom")` - hash_key properly used

   This inconsistency makes the API unpredictable and error-prone.

3. **Documentation Ambiguity**: The docstring states "Hash_key for string key to encode" but doesn't clarify that this only applies to object arrays. The parameter documentation at line 248 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/util/hashing.py` gives no indication of this limitation.

4. **No Error or Warning**: The function provides no feedback when `hash_key` is ignored. Users cannot detect they're passing a useless parameter, potentially leading to security issues if they rely on custom hash keys for data integrity.

5. **Breaks Use Cases**: Users who need deterministic hashing with custom keys across different environments or applications cannot achieve this for numeric data, even though the API suggests this should be possible.

## Relevant Context

The root cause is in the `_hash_ndarray` function (lines 282-339 in `pandas/core/util/hashing.py`):

- Lines 301-306 handle numeric types by converting them to uint64 format but never apply the hash_key
- Line 326 calls `hash_object_array` with the hash_key parameter, but this is only reached for non-numeric object arrays
- The hash_key parameter flows through the entire call stack but is simply never used in the numeric conversion paths

This appears to be an oversight in the implementation rather than intentional design, as:
- The related `hash_pandas_object` function passes hash_key consistently to all `hash_array` calls
- The C extension `hash_object_array` properly uses the hash_key parameter
- There's no performance or technical reason why hash_key couldn't be applied to numeric arrays

Relevant documentation:
- [pandas.util.hash_array documentation](https://pandas.pydata.org/docs/reference/api/pandas.util.hash_array.html)
- Source code: `pandas/core/util/hashing.py` lines 233-279 (hash_array) and 282-339 (_hash_ndarray)

## Proposed Fix

The recommended fix is to apply the hash_key consistently to all array types by mixing it into the numeric values after conversion to uint64:

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -299,11 +299,18 @@ def _hash_ndarray(
     # First, turn whatever array this is into unsigned 64-bit ints, if we can
     # manage it.
     if dtype == bool:
         vals = vals.astype("u8")
     elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
         vals = vals.view("i8").astype("u8", copy=False)
     elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
         vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
+
+    # Apply hash_key to numeric values for consistency
+    if dtype in (bool,) or issubclass(dtype.type, (np.datetime64, np.timedelta64, np.number)):
+        if hash_key != _default_hash_key:
+            # Mix hash_key into the values using XOR with hash of the key
+            from hashlib import sha256
+            key_hash = int.from_bytes(sha256(hash_key.encode('utf8')).digest()[:8], 'little')
+            vals = vals ^ np.uint64(key_hash)
     else:
         # With repeated values, its MUCH faster to categorize object dtypes,
         # then hash and rename categories. We allow skipping the categorization
```