# Bug Report: xarray.core.formatting_html.collapsible_section() Missing HTML Escaping

**Target**: `xarray.core.formatting_html.collapsible_section()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `collapsible_section()` function does not escape HTML special characters in the `name` parameter, making it inconsistent with other similar functions in the same module and violating the principle of defense in depth.

## Property-Based Test

```python
from html import escape
from hypothesis import given, strategies as st, example
from xarray.core.formatting_html import collapsible_section

@given(st.text(min_size=1, max_size=100))
@example("<script>alert('xss')</script>")
@example("Name&Test")
def test_collapsible_section_escapes_name(name):
    """
    Property: collapsible_section should escape HTML special characters in name.
    Evidence: This function takes arbitrary text and embeds it in HTML.
    """
    result = collapsible_section(name)

    if '<script>' in name:
        assert '<script>' not in result, \
            "Unescaped '<script>' found in output"
```

**Failing input**: `"<script>alert('xss')</script>"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import collapsible_section

result = collapsible_section("<script>alert('xss')</script>")
print(result)

assert '<script>' in result
```

**Output**:
```html
<input id='section-...' class='xr-section-summary-in' type='checkbox' disabled >
<label for='section-...' class='xr-section-summary'  title='Expand/collapse section'>
<script>alert('xss')</script>:</label>
<div class='xr-section-inline-details'></div>
<div class='xr-section-details'></div>
```

## Why This Is A Bug

1. **API Inconsistency**: All other similar functions in `formatting_html.py` that handle text parameters use `html.escape()`:
   - `format_dims()` (line 57): `escape(str(dim))`
   - `summarize_attrs()` (line 66): `escape(str(k))` and `escape(str(v))`
   - `summarize_variable()` (line 84-86): escapes name, dims, and dtype
   - `summarize_index()` (line 145): `escape(str(n))`

2. **Contract Violation**: The function accepts a string parameter and embeds it directly in HTML without sanitization.

3. **Current Impact**: Low - the function is currently only called with hardcoded section names ("Coordinates", "Data variables", "Indexes", "Attributes", "Dimensions"), so no actual vulnerability exists in the current codebase.

4. **Future Risk**: If future code passes user-controlled data (e.g., custom section names) to this function, it could introduce an XSS vulnerability.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -175,6 +175,7 @@ def collapsible_section(
 ) -> str:
     # "unique" id to expand/collapse the section
     data_id = "section-" + str(uuid.uuid4())
+    name_escaped = escape(str(name))

     has_items = n_items is not None and n_items
     n_items_span = "" if n_items is None else f" <span>({n_items})</span>"
@@ -185,7 +186,7 @@ def collapsible_section(
     return (
         f"<input id='{data_id}' class='xr-section-summary-in' "
         f"type='checkbox' {enabled} {collapsed}>"
-        f"<label for='{data_id}' class='xr-section-summary' {tip}>"
-        f"{name}:{n_items_span}</label>"
+        f"<label for='{data_id}' class='xr-section-summary'{tip}>"
+        f"{name_escaped}:{n_items_span}</label>"
         f"<div class='xr-section-inline-details'>{inline_details}</div>"
         f"<div class='xr-section-details'>{details}</div>"
     )
```