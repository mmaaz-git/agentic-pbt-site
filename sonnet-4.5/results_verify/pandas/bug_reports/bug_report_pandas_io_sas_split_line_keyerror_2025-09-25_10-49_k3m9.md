# Bug Report: pandas.io.sas _split_line Unconditional KeyError

**Target**: `pandas.io.sas.sas_xport._split_line`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_split_line` helper function unconditionally deletes the `"_"` key from the output dictionary, causing a KeyError if the `parts` parameter doesn't include a tuple with name `"_"`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas_xport import _split_line

@given(
    parts=st.lists(
        st.tuples(
            st.text(st.characters(whitelist_categories=('Lu', 'Ll')),
                   min_size=1, max_size=10).filter(lambda x: x != '_'),
            st.integers(min_value=1, max_value=5)
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=200)
def test_split_line_without_underscore(parts):
    total_length = sum(length for _, length in parts)
    test_string = 'A' * total_length

    try:
        result = _split_line(test_string, parts)
        assert False, f"Should crash when no '_' key present"
    except KeyError as e:
        assert str(e) == "'_'"
```

**Failing input**: `parts=[("first", 2), ("second", 1)]`

## Reproducing the Bug

```python
from pandas.io.sas.sas_xport import _split_line

test_string = "ABC"
parts = [("first", 2), ("second", 1)]

try:
    result = _split_line(test_string, parts)
    print(f"ERROR: Should have raised KeyError, got: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
```

## Why This Is A Bug

The function's docstring says "name '_' will be filtered from output" but doesn't explicitly require it to be present. The code at line 174 unconditionally executes:

```python
del out["_"]
```

This causes a crash if `parts` doesn't contain a `("_", length)` tuple. The function should either:
1. Document that `"_"` must be present in parts, OR
2. Use `out.pop("_", None)` to safely remove it if present

Since the docstring says it "will be filtered", option 2 is more appropriate - the filtering is conditional on its presence.

## Fix

```diff
--- a/pandas/io/sas/sas_xport.py
+++ b/pandas/io/sas/sas_xport.py
@@ -171,7 +171,7 @@ def _split_line(s: str, parts):
     for name, length in parts:
         out[name] = s[start : start + length].strip()
         start += length
-    del out["_"]
+    out.pop("_", None)
     return out
```