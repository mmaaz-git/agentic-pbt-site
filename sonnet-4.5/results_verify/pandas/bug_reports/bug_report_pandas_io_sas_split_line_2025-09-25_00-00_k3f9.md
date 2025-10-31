# Bug Report: pandas.io.sas._split_line Unconditional KeyError

**Target**: `pandas.io.sas.sas_xport._split_line`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_split_line` function unconditionally attempts to delete the key "_" from its output dictionary, causing a `KeyError` if the caller does not include a "_" field in the `parts` parameter. This violates the principle of least surprise and creates an undocumented contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.io.sas.sas_xport import _split_line

@given(
    st.lists(
        st.tuples(
            st.text(min_size=1, max_size=10),
            st.integers(min_value=1, max_value=20)
        ),
        min_size=1,
        max_size=10
    )
)
def test_split_line_without_underscore(parts):
    assume("_" not in [name for name, _ in parts])

    total_len = sum(length for _, length in parts)
    test_string = "x" * total_len

    result = _split_line(test_string, parts)
    assert len(result) == len(parts)
```

**Failing input**: Any parts list without a "_" field, e.g., `[("field1", 5), ("field2", 5)]`

## Reproducing the Bug

```python
from pandas.io.sas.sas_xport import _split_line

parts = [("name", 10), ("value", 10)]
test_string = "hello     world     "

result = _split_line(test_string, parts)
```

**Output**:
```
KeyError: '_'
```

## Why This Is A Bug

The function signature `_split_line(s: str, parts)` suggests it accepts any list of parts, but the implementation unconditionally executes `del out["_"]` at line 174. This creates an implicit, undocumented requirement that all callers must include a "_" field in the parts list.

While all current internal callers do include this field, the function's API contract is violated because:

1. The function signature doesn't indicate the "_" requirement
2. The docstring doesn't mention this requirement
3. The function fails with a cryptic `KeyError` instead of a clear `ValueError`
4. External code or future refactoring could easily trigger this bug

This is inconsistent with good API design principles, even for internal functions.

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

Alternatively, document the requirement in the docstring:

```diff
--- a/pandas/io/sas/sas_xport.py
+++ b/pandas/io/sas/sas_xport.py
@@ -160,6 +160,7 @@ def _split_line(s: str, parts):
     s: str
         Fixed-length string to split
     parts: list of (name, length) pairs
         Used to break up string, name '_' will be filtered from output.
+        Note: parts MUST include at least one ('_', length) entry.

     Returns
```