# Bug Report: xarray collapsible_section Negative n_items Logic Error

**Target**: `xarray.core.formatting_html.collapsible_section`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `collapsible_section` function incorrectly treats negative `n_items` values as truthy, causing sections to appear enabled with negative item counts displayed instead of being disabled.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.formatting_html import collapsible_section

@given(
    st.text(min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=-1)
)
def test_collapsible_section_negative_n_items(name, n_items):
    """Property: collapsible_section should treat negative n_items as invalid"""
    result = collapsible_section(name, n_items=n_items)

    assert "disabled" in result, "Negative n_items should result in disabled section"
```

**Failing input**: `name='test'`, `n_items=-1`

## Reproducing the Bug

```python
from xarray.core.formatting_html import collapsible_section

result = collapsible_section("Test Section", n_items=-1)

print("Contains 'disabled':", "disabled" in result)
print("Contains 'checked':", "checked" in result)
print("Displays count:", "(-1)" in result)
```

**Output:**
```
Contains 'disabled': False
Contains 'checked': True
Displays count: True
```

**Expected behavior:** Section should be disabled when n_items is negative (same as n_items=0 or None).

## Why This Is A Bug

The function uses `has_items = n_items is not None and n_items` to check if there are items to display. In Python, negative integers are truthy, so `-1` evaluates to `True` in a boolean context. This causes:

1. The section to be enabled when it should be disabled
2. The checkbox to have `checked` attribute when it shouldn't
3. A nonsensical negative count `(-1)` to be displayed

While callers typically pass `len(mapping)` (always non-negative), the function's public API doesn't validate input and can be called directly with invalid values.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -176,7 +176,7 @@ def collapsible_section(
     # "unique" id to expand/collapse the section
     data_id = "section-" + str(uuid.uuid4())

-    has_items = n_items is not None and n_items
+    has_items = n_items is not None and n_items > 0
     n_items_span = "" if n_items is None else f" <span>({n_items})</span>"
     enabled = "" if enabled and has_items else "disabled"
     collapsed = "" if collapsed or not has_items else "checked"
```

This change ensures that only positive item counts are treated as "having items", making negative values behave the same as zero or None.