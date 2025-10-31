# Bug Report: pandas.io.sas._split_line KeyError on Missing Underscore

**Target**: `pandas.io.sas.sas_xport._split_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_split_line` function unconditionally deletes the "_" key from its output dictionary, causing a KeyError when the input `parts` list doesn't contain an entry named "_".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas_xport import _split_line


@given(
    st.lists(
        st.tuples(
            st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1, max_size=10),
            st.integers(min_value=1, max_value=20)
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=1000)
def test_split_line_key_invariant(parts):
    total_length = sum(length for _, length in parts)
    s = 'x' * total_length

    result = _split_line(s, parts)

    expected_keys = {name for name, _ in parts if name != '_'}
    assert set(result.keys()) == expected_keys
```

**Failing input**: `parts=[('name', 5), ('value', 3)]`, `s='Alice123'`

## Reproducing the Bug

```python
from pandas.io.sas.sas_xport import _split_line

parts = [('name', 5), ('value', 3)]
s = 'Alice123'

result = _split_line(s, parts)
```

Output:
```
KeyError: '_'
```

## Why This Is A Bug

The function's docstring states it splits a fixed-length string based on a list of (name, length) pairs and filters out entries named "_". However, the implementation unconditionally executes `del out["_"]` at line 174, which assumes "_" always exists in the parts list.

While current usage in the codebase always includes "_" in parts (e.g., lines 307, 336, 338), the function is defined as a general utility that doesn't document this requirement. This makes the code fragile and could cause crashes if used in different contexts or during refactoring.

## Fix

```diff
--- a/pandas/io/sas/sas_xport.py
+++ b/pandas/io/sas/sas_xport.py
@@ -171,7 +171,8 @@ def _split_line(s: str, parts):
     for name, length in parts:
         out[name] = s[start : start + length].strip()
         start += length
-    del out["_"]
+    if "_" in out:
+        del out["_"]
     return out
```
