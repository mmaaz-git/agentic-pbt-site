# Bug Report: pandas.io.json._normalize.nested_to_record KeyError with Non-String Keys

**Target**: `pandas.io.json._normalize.nested_to_record`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `nested_to_record` function crashes with a `KeyError` when processing nested dictionaries that have non-string keys. The function converts non-string keys to strings but then attempts to use the stringified key to access the original dictionary, which still has the non-string key.

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
```

**Failing input**: `{1: {2: 'value'}}`

## Reproducing the Bug

```python
from pandas.io.json._normalize import nested_to_record

d = {1: {2: 'value'}}
result = nested_to_record(d)
```

Output:
```
KeyError: '1'
```

More complex example:
```python
d = {1: 'a', 2: 'b', 3: {4: 'c', 5: 'd'}}
result = nested_to_record(d)
```

Output:
```
KeyError: '3'
```

## Why This Is A Bug

The function is documented to accept "dict or list of dicts" without any restriction that keys must be strings. In JSON, while keys are typically strings, Python dictionaries can have any hashable type as keys, and the function should handle this gracefully.

The bug occurs in this code sequence:

```python
for k, v in d.items():
    if not isinstance(k, str):
        k = str(k)  # Convert k to string

    # ... later ...

    v = new_d.pop(k)  # BUG: tries to pop using stringified key
```

When the original key is an integer (e.g., `3`), the code converts it to `"3"` (string), but `new_d` still contains the integer key `3`. Attempting `new_d.pop("3")` raises a `KeyError` because the key `"3"` doesn't exist in the dictionary.

## Fix

The fix is to preserve the original key for dictionary operations and only use the stringified version for the new key name:

```diff
--- a/pandas/io/json/_normalize.py
+++ b/pandas/io/json/_normalize.py
@@ -103,9 +103,10 @@ def nested_to_record(
         new_d = copy.deepcopy(d)
         for k, v in d.items():
             # each key gets renamed with prefix
+            original_k = k
             if not isinstance(k, str):
                 k = str(k)
             if level == 0:
                 newkey = k
             else:
                 newkey = prefix + sep + k
@@ -114,11 +115,11 @@ def nested_to_record(
             # current dict level  < maximum level provided and
             # only dicts gets recurse-flattened
             # only at level>1 do we rename the rest of the keys
             if not isinstance(v, dict) or (
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