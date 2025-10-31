# Bug Report: xarray.core.formatting_html._icon Missing HTML Escaping

**Target**: `xarray.core.formatting_html._icon`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_icon` function fails to HTML-escape its `icon_name` parameter before inserting it into SVG HTML, violating the expected contract for HTML generation functions.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from html import escape
from xarray.core.formatting_html import _icon

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1))
def test_icon_escapes_name(icon_name):
    result = _icon(icon_name)
    escaped_icon = escape(str(icon_name))
    if escaped_icon != icon_name:
        assert escaped_icon in result, "Icon name should be HTML-escaped if needed"
```

**Failing input**: `icon_name="'"`

## Reproducing the Bug

```python
from xarray.core.formatting_html import _icon

icon_name = "test'><script>alert('XSS')</script><svg class='"
result = _icon(icon_name)

print("Generated HTML:", result)
print("Contains unescaped input:", "'" in result)
```

## Why This Is A Bug

HTML generation functions should escape their string parameters to prevent malformed HTML and potential injection vulnerabilities. While current callers only pass hardcoded icon names like "icon-database", the function's API implies it should handle arbitrary strings safely. The lack of escaping creates broken HTML when the icon name contains special characters.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -73,7 +73,8 @@ def _icon(icon_name) -> str:
     # icon_name should be defined in xarray/static/html/icon-svg-inline.html
     return (
-        f"<svg class='icon xr-{icon_name}'><use xlink:href='#{icon_name}'></use></svg>"
+        f"<svg class='icon xr-{escape(str(icon_name))}'>"
+        f"<use xlink:href='#{escape(str(icon_name))}'></use></svg>"
     )
```