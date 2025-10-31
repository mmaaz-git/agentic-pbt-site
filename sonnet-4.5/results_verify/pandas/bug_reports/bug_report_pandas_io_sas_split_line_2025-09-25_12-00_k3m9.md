# Bug Report: pandas.io.sas._split_line KeyError on Missing '_' Field

**Target**: `pandas.io.sas.sas_xport._split_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_split_line` function in `pandas.io.sas.sas_xport` crashes with a `KeyError` when called with a `parts` list that doesn't contain a field named `"_"`. The function unconditionally attempts to delete `out["_"]` without checking if this key exists.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.sas.sas_xport import _split_line

@given(
    st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20).filter(lambda s: s != "_"),
            st.integers(min_value=1, max_value=50)
        ),
        min_size=1,
        max_size=20
    )
)
def test_split_line_no_underscore_field(parts):
    total_length = sum(length for _, length in parts)
    test_string = "x" * total_length

    result = _split_line(test_string, parts)
```

**Failing input**: `parts=[('field', 4)]`, `s='test'`

## Reproducing the Bug

```python
from pandas.io.sas.sas_xport import _split_line

parts = [('field', 4)]
s = 'test'
result = _split_line(s, parts)
```

**Output**:
```
KeyError: '_'
```

## Why This Is A Bug

The `_split_line` function is designed to parse fixed-width strings into dictionaries based on field specifications. The function uses `"_"` as a special field name to indicate ignored portions of the string. However, the function unconditionally executes `del out["_"]` at line 174, even when no `"_"` field exists in the input.

This violates the principle of least surprise - a user providing a valid-looking `parts` parameter (a list of name-length tuples) would not expect the function to require a specific field name. The function should either:
1. Make the `"_"` field requirement explicit in the signature/documentation
2. Check if `"_"` exists before attempting to delete it
3. Use a different mechanism to handle ignored fields

Currently, all internal calls to `_split_line` in pandas.io.sas include a `"_"` field, so this bug is not triggered in practice. However, this represents fragile API design that could break if the function is reused elsewhere or if the calling code changes.

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