# Bug Report: xarray.core.formatting_html.collapsible_section Incorrect Tooltip Logic

**Target**: `xarray.core.formatting_html.collapsible_section`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `collapsible_section` function has inverted logic for the tooltip: it shows "Expand/collapse section" tooltip on disabled checkboxes (where interaction is impossible) but omits it on enabled checkboxes (where it would be helpful).

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(
    st.text(min_size=1),
    st.text(),
    st.text(),
    st.integers(min_value=0, max_value=10),
    st.booleans(),
    st.booleans()
)
def test_tooltip_should_match_enabled_state(name, inline_details, details, n_items, enabled_param, collapsed):
    """Tooltip should appear on enabled checkboxes, not disabled ones"""
    result = collapsible_section(name, inline_details, details, n_items, enabled_param, collapsed)

    has_items = n_items is not None and n_items
    is_enabled = enabled_param and has_items
    has_tooltip = "title='Expand/collapse section'" in result

    if is_enabled:
        assert has_tooltip, "Enabled checkboxes should have tooltip"
    else:
        assert not has_tooltip, "Disabled checkboxes should not have tooltip"
```

**Failing input**: `name="Coords", n_items=0, enabled=True` (and many others)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import collapsible_section

result_enabled = collapsible_section("Coords", "", "", n_items=5, enabled=True)
result_disabled = collapsible_section("Dims", "", "", n_items=0, enabled=True)

print("Enabled checkbox (n_items=5):")
print(f"  Has tooltip: {'title=' in result_enabled}")
print(f"  Has 'disabled': {'disabled' in result_enabled}")

print("\nDisabled checkbox (n_items=0):")
print(f"  Has tooltip: {'title=' in result_disabled}")
print(f"  Has 'disabled': {'disabled' in result_disabled}")
```

**Output:**
```
Enabled checkbox (n_items=5):
  Has tooltip: False  ← Should be True
  Has 'disabled': False

Disabled checkbox (n_items=0):
  Has tooltip: True  ← Should be False
  Has 'disabled': True
```

## Why This Is A Bug

The root cause is on line 183 of `formatting_html.py`:

```python
enabled = "" if enabled and has_items else "disabled"  # Line 181: reassigns to string
tip = " title='Expand/collapse section'" if enabled else ""  # Line 183: checks the string!
```

After line 181, `enabled` is a string (`""` or `"disabled"`), not a boolean:
- `""` (empty string) is falsy → no tooltip added
- `"disabled"` (non-empty string) is truthy → tooltip added

This is backwards: disabled checkboxes get the "Expand/collapse" tooltip, while enabled ones don't.

The bug affects real usage. In `dim_section()` at line 230, `collapsible_section` is called with `enabled=False`, producing a disabled checkbox with a misleading tooltip encouraging interaction.

## Fix

```diff
def collapsible_section(
    name, inline_details="", details="", n_items=None, enabled=True, collapsed=False
) -> str:
    data_id = "section-" + str(uuid.uuid4())

    has_items = n_items is not None and n_items
    n_items_span = "" if n_items is None else f" <span>({n_items})</span>"
+   is_enabled = enabled and has_items
-   enabled = "" if enabled and has_items else "disabled"
+   enabled_attr = "" if is_enabled else "disabled"
    collapsed = "" if collapsed or not has_items else "checked"
-   tip = " title='Expand/collapse section'" if enabled else ""
+   tip = " title='Expand/collapse section'" if is_enabled else ""

    return (
-       f"<input id='{data_id}' class='xr-section-summary-in' "
-       f"type='checkbox' {enabled} {collapsed}>"
+       f"<input id='{data_id}' class='xr-section-summary-in' "
+       f"type='checkbox' {enabled_attr} {collapsed}>"
        f"<label for='{data_id}' class='xr-section-summary' {tip}>"
        f"{name}:{n_items_span}</label>"
        f"<div class='xr-section-inline-details'>{inline_details}</div>"
        f"<div class='xr-section-details'>{details}</div>"
    )
```

The fix saves the boolean state in `is_enabled` before converting to the HTML attribute string, then uses the boolean for the tooltip logic.