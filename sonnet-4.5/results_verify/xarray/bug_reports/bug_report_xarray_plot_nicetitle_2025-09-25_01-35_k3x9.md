# Bug Report: xarray.plot._nicetitle Truncation Violates maxchar Constraint

**Target**: `xarray.plot.facetgrid._nicetitle`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_nicetitle` function violates its documented behavior by returning strings that exceed the `maxchar` parameter when `maxchar < 4`. The function's docstring states it should "truncate at maxchar", but it can return strings up to 3 characters longer than specified.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.plot.facetgrid import _nicetitle

@given(
    st.text(min_size=0, max_size=100),
    st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=100)),
    st.integers(min_value=1, max_value=1000)
)
def test_nicetitle_max_length_property(coord, value, maxchar):
    template = "{coord}={value}"
    result = _nicetitle(coord, value, maxchar, template)
    assert len(result) <= maxchar
```

**Failing input**: `coord='', value=0, maxchar=1`

## Reproducing the Bug

```python
from xarray.plot.facetgrid import _nicetitle

result = _nicetitle('', 0, maxchar=1, template="{coord}={value}")
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")

assert len(result) <= 1
```

Output:
```
Result: '...'
Length: 3
AssertionError
```

## Why This Is A Bug

The function's docstring explicitly states: "Put coord, value in template and truncate at maxchar". This clearly implies that the returned string should have length at most `maxchar`. However, when `maxchar < 4`, the function returns "..." (length 3), which violates this constraint.

The bug occurs because the implementation at line 54 does:
```python
title = title[: (maxchar - 3)] + "..."
```

This logic assumes that `maxchar >= 3` to accommodate the ellipsis. When `maxchar < 3`, the slice becomes negative or zero, resulting in just "..." being returned.

## Fix

```diff
--- a/xarray/plot/facetgrid.py
+++ b/xarray/plot/facetgrid.py
@@ -51,7 +51,11 @@ def _nicetitle(coord, value, maxchar, template):
     title = template.format(coord=coord, value=prettyvalue)

     if len(title) > maxchar:
-        title = title[: (maxchar - 3)] + "..."
+        if maxchar < 3:
+            title = title[:maxchar]
+        else:
+            title = title[: (maxchar - 3)] + "..."

     return title
```