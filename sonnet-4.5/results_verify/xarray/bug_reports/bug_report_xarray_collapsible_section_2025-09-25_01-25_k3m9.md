# Bug Report: xarray.core.formatting_html.collapsible_section Missing HTML Escaping

**Target**: `xarray.core.formatting_html.collapsible_section`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `collapsible_section` function fails to HTML-escape its `name` parameter before inserting it into the generated HTML, violating the expected contract for HTML generation functions.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from html import escape
from xarray.core.formatting_html import collapsible_section

@given(st.text(min_size=1, max_size=50))
def test_collapsible_section_escapes_name(name):
    result = collapsible_section(name)
    escaped_name = escape(str(name))
    assert escaped_name in result, "Section name should be HTML-escaped"
```

**Failing input**: `name="'"`

## Reproducing the Bug

```python
from xarray.core.formatting_html import collapsible_section
from html import escape

name = "<script>alert('XSS')</script>"
result = collapsible_section(name)

print("Generated HTML:", result)
print("Contains unescaped input:", name in result)
print("Expected escaped:", escape(name))
```

## Why This Is A Bug

HTML generation functions should escape their string parameters to prevent malformed HTML and potential injection vulnerabilities. While current callers only pass hardcoded strings, the function's API contract implies it should handle arbitrary strings safely. The function escapes other content (via helper functions) but inconsistently leaves the `name` parameter unescaped.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -185,7 +185,7 @@ def collapsible_section(
     return (
         f"<input id='{data_id}' class='xr-section-summary-in' "
         f"type='checkbox' {enabled} {collapsed}>"
-        f"<label for='{data_id}' class='xr-section-summary' {tip}>"
-        f"{name}:{n_items_span}</label>"
+        f"<label for='{data_id}' class='xr-section-summary'{tip}>"
+        f"{escape(str(name))}:{n_items_span}</label>"
         f"<div class='xr-section-inline-details'>{inline_details}</div>"
         f"<div class='xr-section-details'>{details}</div>"
     )
```