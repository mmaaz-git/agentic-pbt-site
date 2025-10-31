# Bug Report: xarray.core.formatting_html.collapsible_section Incorrect Handling of Negative n_items

**Target**: `xarray.core.formatting_html.collapsible_section`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `collapsible_section` function treats negative `n_items` values as truthy, resulting in sections being enabled and expanded when they should be disabled. While this doesn't occur in the current codebase (since `n_items` is derived from `len()`), the function doesn't validate its input and behaves incorrectly with negative values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.core.formatting_html import collapsible_section

@given(st.integers(min_value=-100, max_value=-1))
@settings(max_examples=200)
def test_collapsible_section_negative_n_items_should_be_disabled(n_items):
    result = collapsible_section("Test", "", "", n_items=n_items, enabled=True, collapsed=False)

    assert "disabled" in result, \
        f"Negative n_items={n_items} should result in disabled section"
    assert "checked" not in result, \
        f"Negative n_items={n_items} should not result in checked/expanded section"
```

**Failing input**: `n_items=-5`

## Reproducing the Bug

```python
from xarray.core.formatting_html import collapsible_section

result = collapsible_section(
    name="Test Section",
    n_items=-5,
    enabled=True,
    collapsed=False
)

print(result)

has_items = -5 is not None and -5
print(f"has_items evaluates to: {has_items}")
print(f"bool(has_items) = {bool(has_items)}")
```

Output shows:
- `has_items = -5` (truthy!)
- Section is NOT disabled
- Section displays `(-5)` which is semantically meaningless
- Section is checked/expanded

## Why This Is A Bug

A negative item count is semantically meaningless. The function should treat `n_items <= 0` the same way it treats `n_items = 0` or `n_items = None` - the section should be disabled and not expanded.

The root cause is line 179 in `formatting_html.py`:

```python
has_items = n_items is not None and n_items
```

When `n_items` is a negative integer, `n_items is not None` is `True`, and `True and -5` evaluates to `-5`, which is truthy. This causes:
- Line 181: `enabled = ""` (not "disabled") because `enabled and has_items` is truthy
- Line 182: `collapsed = "checked"` because `not has_items` is `False`

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

This ensures that only positive item counts are treated as "having items", making the behavior consistent with the semantic meaning of the parameter.