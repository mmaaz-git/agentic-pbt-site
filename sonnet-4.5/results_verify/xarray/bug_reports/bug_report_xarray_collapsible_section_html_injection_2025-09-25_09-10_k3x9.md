# Bug Report: xarray.core.formatting_html.collapsible_section HTML Injection

**Target**: `xarray.core.formatting_html.collapsible_section`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `collapsible_section` function in `xarray.core.formatting_html` does not escape the `name` parameter, allowing HTML injection when user-controlled input is passed. This is inconsistent with other functions in the same module that properly escape user input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.formatting_html import collapsible_section

@given(st.text())
def test_collapsible_section_escapes_html_in_name(user_input):
    html = collapsible_section(user_input)
    if '<script>' in user_input:
        assert '<script>' not in html or '&lt;script&gt;' in html
```

**Failing input**: `'<script>alert("XSS")</script>'`

## Reproducing the Bug

```python
from xarray.core.formatting_html import collapsible_section

user_input = '<script>alert("XSS")</script>'
html = collapsible_section(name=user_input)

print(html)

assert '<script>' in html
assert '&lt;script&gt;' not in html
```

## Why This Is A Bug

1. **Inconsistency**: Other functions in the same module (e.g., `format_dims`, `summarize_attrs`) use `escape()` to sanitize user input before including it in HTML
2. **Security risk**: If external code or plugins call this public function with user-controlled input, it could lead to XSS vulnerabilities
3. **API contract violation**: Users reasonably expect that HTML-generating functions will escape dangerous characters unless explicitly documented otherwise

While internal callers currently use hardcoded strings for the `name` parameter, the function is publicly exported and could be called by external code.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -173,7 +173,7 @@ def collapsible_section(
     collapsed = "" if collapsed or not has_items else "checked"
     tip = " title='Expand/collapse section'" if enabled else ""

     return (
         f"<input id='{data_id}' class='xr-section-summary-in' "
         f"type='checkbox' {enabled} {collapsed}>"
         f"<label for='{data_id}' class='xr-section-summary' {tip}>"
-        f"{name}:{n_items_span}</label>"
+        f"{escape(name)}:{n_items_span}</label>"
         f"<div class='xr-section-inline-details'>{inline_details}</div>"
         f"<div class='xr-section-details'>{details}</div>"
     )
```