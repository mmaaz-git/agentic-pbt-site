# Bug Report: xarray.plot.facetgrid._nicetitle Maximum Length Violation

**Target**: `xarray.plot.facetgrid._nicetitle`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_nicetitle` function violates its contract by returning strings longer than the specified `maxchar` parameter when `maxchar < 3`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.plot.facetgrid import _nicetitle

@given(
    st.text(min_size=0, max_size=50),
    st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(min_size=0, max_size=100), st.none()),
    st.integers(min_value=1, max_value=1000),
    st.text(min_size=0, max_size=100)
)
@settings(max_examples=1000)
def test_nicetitle_length_property(coord, value, maxchar, template_base):
    template = template_base + "{coord}={value}"
    result = _nicetitle(coord, value, maxchar, template)
    assert len(result) <= maxchar
```

**Failing input**: `coord='', value=None, maxchar=1, template='{coord}={value}'`

## Reproducing the Bug

```python
from xarray.plot.facetgrid import _nicetitle

result = _nicetitle(coord='', value=None, maxchar=1, template='{coord}={value}')
print(f"Result: '{result}'")
print(f"Length: {len(result)}")
print(f"Expected max length: 1")

result2 = _nicetitle(coord='x', value=123, maxchar=2, template='{coord}={value}')
print(f"Result2: '{result2}'")
print(f"Length: {len(result2)}")
print(f"Expected max length: 2")
```

Output:
```
Result: '=No...'
Length: 6
Expected max length: 1

Result2: 'x=12...'
Length: 7
Expected max length: 2
```

## Why This Is A Bug

The function's docstring states it should "Put coord, value in template and truncate at maxchar", implying the output should never exceed `maxchar` characters. However, when `maxchar < 3`, the truncation logic `title[:(maxchar - 3)] + "..."` produces output longer than `maxchar`.

## Fix

```diff
--- a/facetgrid.py
+++ b/facetgrid.py
@@ -46,11 +46,13 @@ def _nicetitle(coord, value, maxchar, template):
 def _nicetitle(coord, value, maxchar, template):
     """
     Put coord, value in template and truncate at maxchar
     """
     prettyvalue = format_item(value, quote_strings=False)
     title = template.format(coord=coord, value=prettyvalue)

     if len(title) > maxchar:
-        title = title[: (maxchar - 3)] + "..."
+        if maxchar >= 3:
+            title = title[: (maxchar - 3)] + "..."
+        else:
+            title = title[:maxchar]

     return title
```