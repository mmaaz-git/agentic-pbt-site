# Bug Report: pandas.io.json._normalize.nested_to_record KeyError with Non-String Dictionary Keys

**Target**: `pandas.io.json._normalize.nested_to_record`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `nested_to_record` function crashes with a `KeyError` when processing nested dictionaries that contain non-string keys. The function converts non-string keys to strings internally but then attempts to access the dictionary using the stringified key while the dictionary still contains the original non-string key.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.json._normalize import nested_to_record

@given(st.dictionaries(
    keys=st.integers(),
    values=st.one_of(
        st.text(),
        st.dictionaries(keys=st.integers(), values=st.text(), max_size=3)
    ),
    max_size=5
))
def test_nested_to_record_handles_non_string_keys(d):
    result = nested_to_record(d)
    assert isinstance(result, dict)

# Run the test
if __name__ == "__main__":
    test_nested_to_record_handles_non_string_keys()
```

<details>

<summary>
**Failing input**: `{0: {}}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 18, in <module>
    test_nested_to_record_handles_non_string_keys()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 5, in test_nested_to_record_handles_non_string_keys
    keys=st.integers(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 13, in test_nested_to_record_handles_non_string_keys
    result = nested_to_record(d)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_normalize.py", line 117, in nested_to_record
    v = new_d.pop(k)
KeyError: '0'
Falsifying example: test_nested_to_record_handles_non_string_keys(
    d={0: {}},
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_normalize.py:110
```
</details>

## Reproducing the Bug

```python
from pandas.io.json._normalize import nested_to_record

# Test case that crashes
d = {1: {2: 'value'}}
try:
    result = nested_to_record(d)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()

# More complex example
d2 = {1: 'a', 2: 'b', 3: {4: 'c', 5: 'd'}}
try:
    result2 = nested_to_record(d2)
    print(f"Result: {result2}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
KeyError when processing nested dictionaries with integer keys
</summary>
```
Error: KeyError: '1'

Error: KeyError: '3'
```
</details>

## Why This Is A Bug

The function's docstring explicitly states it accepts "dict or list of dicts" without any restriction that keys must be strings. The code at lines 98-99 of `_normalize.py` demonstrates clear intent to handle non-string keys by converting them to strings:

```python
if not isinstance(k, str):
    k = str(k)
```

However, there's a logic error in the implementation. The bug occurs because:

1. The code iterates over the original dictionary items (line 96)
2. When it encounters a non-string key, it converts the key variable `k` to a string (lines 98-99)
3. Later, when the value is a nested dictionary, it attempts to pop the value using `new_d.pop(k)` (lines 113 and 117)
4. Since `k` is now the stringified version (e.g., `"1"`), but `new_d` still contains the original integer key (e.g., `1`), the pop operation fails with a `KeyError`

The function works correctly for flat dictionaries with non-string keys, but crashes when those keys have nested dictionary values. This inconsistency and the presence of code attempting to handle non-string keys indicates this is a bug rather than intentional behavior.

## Relevant Context

- The bug only occurs when non-string keys have nested dictionary values
- Flat dictionaries with non-string keys work correctly
- The function is commonly used in data processing pipelines where Python dictionaries with integer or other hashable keys are common
- The pandas JSON normalization functions are widely used for flattening nested data structures, not just JSON data
- Source code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/json/_normalize.py:113,117`

## Proposed Fix

The fix is to preserve the original key for dictionary operations while using the stringified version only for creating new keys:

```diff
--- a/pandas/io/json/_normalize.py
+++ b/pandas/io/json/_normalize.py
@@ -94,6 +94,7 @@ def nested_to_record(
     for d in ds:
         new_d = copy.deepcopy(d)
         for k, v in d.items():
+            original_k = k
             # each key gets renamed with prefix
             if not isinstance(k, str):
                 k = str(k)
@@ -110,11 +111,11 @@ def nested_to_record(
                 max_level is not None and level >= max_level
             ):
                 if level != 0:  # so we skip copying for top level, common case
-                    v = new_d.pop(k)
+                    v = new_d.pop(original_k)
                     new_d[newkey] = v
                 continue

-            v = new_d.pop(k)
+            v = new_d.pop(original_k)
             new_d.update(nested_to_record(v, newkey, sep, level + 1, max_level))
         new_ds.append(new_d)
```