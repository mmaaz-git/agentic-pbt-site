# Bug Report: xarray.core.formatting_html._icon() Missing HTML Escaping

**Target**: `xarray.core.formatting_html._icon()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_icon()` function does not escape HTML special characters in the `icon_name` parameter, making it inconsistent with other similar functions in the same module and violating the principle of defense in depth.

## Property-Based Test

```python
from html import escape
from hypothesis import given, strategies as st, example
from xarray.core.formatting_html import _icon

@given(st.text(min_size=1, max_size=100))
@example("<script>alert('xss')</script>")
@example("test&name")
def test_icon_escapes_html_characters(icon_name):
    """
    Property: _icon should escape HTML special characters in icon_name.
    Evidence: Other functions in the same file use escape() for user inputs.
    """
    result = _icon(icon_name)

    dangerous_chars = ['<', '>', '&', '"', "'"]
    if any(char in icon_name for char in dangerous_chars):
        escaped_icon_name = escape(str(icon_name))
        assert escaped_icon_name in result or icon_name not in result, \
            f"Icon name should be HTML-escaped"
```

**Failing input**: `"<script>alert('xss')</script>"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _icon

result = _icon("<script>alert('xss')</script>")
print(result)

assert '<script>' in result
```

**Output**:
```html
<svg class='icon xr-<script>alert('xss')</script>'><use xlink:href='#<script>alert('xss')</script>'></use></svg>
```

## Why This Is A Bug

1. **API Inconsistency**: All other similar functions in `formatting_html.py` that handle text parameters use `html.escape()`:
   - `format_dims()` (line 57): `escape(str(dim))`
   - `summarize_attrs()` (line 66): `escape(str(k))` and `escape(str(v))`
   - `summarize_variable()` (line 84-86): escapes name, dims, and dtype

2. **Contract Violation**: The function accepts arbitrary strings but doesn't sanitize them before embedding in HTML, violating expectations.

3. **Current Impact**: Low - the function is currently only called with hardcoded icon names ("icon-file-text2", "icon-database"), so no actual vulnerability exists in the current codebase.

4. **Future Risk**: If future code passes user-controlled data to this function, it could introduce an XSS vulnerability.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -72,7 +72,7 @@ def summarize_attrs(attrs) -> str:

 def _icon(icon_name) -> str:
     # icon_name should be defined in xarray/static/html/icon-svg-inline.html
+    icon_name_escaped = escape(str(icon_name))
     return (
-        f"<svg class='icon xr-{icon_name}'><use xlink:href='#{icon_name}'></use></svg>"
+        f"<svg class='icon xr-{icon_name_escaped}'><use xlink:href='#{icon_name_escaped}'></use></svg>"
     )
```