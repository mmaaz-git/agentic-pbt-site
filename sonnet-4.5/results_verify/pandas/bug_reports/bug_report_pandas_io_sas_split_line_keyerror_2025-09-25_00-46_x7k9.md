# Bug Report: pandas.io.sas _split_line KeyError

**Target**: `pandas.io.sas.sas_xport._split_line`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_split_line` function unconditionally executes `del out["_"]` regardless of whether the `parts` parameter contains a tuple with name `"_"`. This causes a KeyError when the function is called with inputs that don't include underscore placeholders.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
from pandas.io.sas.sas_xport import _split_line

@given(
    parts=st.lists(
        st.tuples(
            st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=1, max_size=10),
            st.integers(min_value=1, max_value=20)
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=500)
def test_split_line_handles_missing_underscore(parts):
    parts_without_underscore = [(name, length) for name, length in parts if name != "_"]
    assume(len(parts_without_underscore) > 0)

    total_length = sum(length for _, length in parts_without_underscore)
    test_string = "A" * total_length

    result = _split_line(test_string, parts_without_underscore)

    assert isinstance(result, dict)
    assert "_" not in result
```

**Failing input**: `parts = [('0', 1)]`

## Reproducing the Bug

```python
from pandas.io.sas.sas_xport import _split_line

test_string = "A"
parts_without_underscore = [("field1", 1)]

try:
    result = _split_line(test_string, parts_without_underscore)
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print("Bug confirmed: _split_line crashes when '_' is not in parts")
```

## Why This Is A Bug

The function's docstring states that `name '_' will be filtered from output`, which implies that "_" is optional. However, the implementation assumes "_" is always present in the parts list. While all current callers in the codebase do include "_" in their parts lists, this is a design flaw that:

1. Violates the function's implicit contract (optional filtering)
2. Makes the function fragile for future use
3. Could cause crashes if the function is ever reused with different inputs

The bug is currently dormant because all existing callers include "_" in their parts:
- Line 306: `fif = [["prefix", 24], ["version", 8], ["OS", 8], ["_", 24], ["created", 16]]`
- Line 333: `mem = [..., ["_", 24], ...]`
- Line 337: `mem = [["modified", 16], ["_", 16], ["label", 40], ["type", 8]]`

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