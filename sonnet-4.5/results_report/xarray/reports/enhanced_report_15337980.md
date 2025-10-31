# Bug Report: xarray.core.formatting_html.collapsible_section Inverted Tooltip Logic

**Target**: `xarray.core.formatting_html.collapsible_section`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `collapsible_section` function displays tooltips on disabled checkboxes (which cannot be interacted with) but omits tooltips on enabled checkboxes (where user interaction is possible), due to a variable type confusion bug.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.core.formatting_html import collapsible_section

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
        assert has_tooltip, f"Enabled checkboxes should have tooltip. Params: name={name!r}, n_items={n_items}, enabled={enabled_param}, collapsed={collapsed}"
    else:
        assert not has_tooltip, f"Disabled checkboxes should not have tooltip. Params: name={name!r}, n_items={n_items}, enabled={enabled_param}, collapsed={collapsed}"

# Run the test
test_tooltip_should_match_enabled_state()
```

<details>

<summary>
**Failing input**: `name='0', n_items=1, enabled_param=True` (enabled checkbox without tooltip)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 29, in <module>
  |     test_tooltip_should_match_enabled_state()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 8, in test_tooltip_should_match_enabled_state
  |     st.text(min_size=1),
  |                ^^^
  |   File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 24, in test_tooltip_should_match_enabled_state
    |     assert has_tooltip, f"Enabled checkboxes should have tooltip. Params: name={name!r}, n_items={n_items}, enabled={enabled_param}, collapsed={collapsed}"
    |            ^^^^^^^^^^^
    | AssertionError: Enabled checkboxes should have tooltip. Params: name='0', n_items=1, enabled=True, collapsed=False
    | Falsifying example: test_tooltip_should_match_enabled_state(
    |     # The test always failed when commented parts were varied together.
    |     name='0',  # or any other generated value
    |     inline_details='',  # or any other generated value
    |     details='',  # or any other generated value
    |     n_items=1,  # or any other generated value
    |     enabled_param=True,  # or any other generated value
    |     collapsed=False,  # or any other generated value
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 26, in test_tooltip_should_match_enabled_state
    |     assert not has_tooltip, f"Disabled checkboxes should not have tooltip. Params: name={name!r}, n_items={n_items}, enabled={enabled_param}, collapsed={collapsed}"
    |            ^^^^^^^^^^^^^^^
    | AssertionError: Disabled checkboxes should not have tooltip. Params: name='0', n_items=0, enabled=False, collapsed=False
    | Falsifying example: test_tooltip_should_match_enabled_state(
    |     # The test always failed when commented parts were varied together.
    |     name='0',  # or any other generated value
    |     inline_details='',  # or any other generated value
    |     details='',  # or any other generated value
    |     n_items=0,  # or any other generated value
    |     enabled_param=False,  # or any other generated value
    |     collapsed=False,  # or any other generated value
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import collapsible_section

# Test with enabled checkbox (should have tooltip but doesn't)
result_enabled = collapsible_section("Coords", "", "", n_items=5, enabled=True)
print("Enabled checkbox (n_items=5, enabled=True):")
print(f"  HTML output contains tooltip: {'title=' in result_enabled}")
print(f"  HTML output contains 'disabled': {'disabled' in result_enabled}")
print()

# Test with disabled checkbox (shouldn't have tooltip but does)
result_disabled = collapsible_section("Dims", "", "", n_items=0, enabled=True)
print("Disabled checkbox (n_items=0, enabled=True):")
print(f"  HTML output contains tooltip: {'title=' in result_disabled}")
print(f"  HTML output contains 'disabled': {'disabled' in result_disabled}")
print()

# Show the actual HTML for clarity
print("Actual HTML for enabled checkbox:")
print(result_enabled)
print()
print("Actual HTML for disabled checkbox:")
print(result_disabled)
```

<details>

<summary>
Output demonstrates inverted tooltip behavior
</summary>
```
Enabled checkbox (n_items=5, enabled=True):
  HTML output contains tooltip: False
  HTML output contains 'disabled': False

Disabled checkbox (n_items=0, enabled=True):
  HTML output contains tooltip: True
  HTML output contains 'disabled': True

Actual HTML for enabled checkbox:
<input id='section-967b872f-c92d-45ea-b4fa-d7075ea26b7b' class='xr-section-summary-in' type='checkbox'  checked><label for='section-967b872f-c92d-45ea-b4fa-d7075ea26b7b' class='xr-section-summary' >Coords: <span>(5)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'></div>

Actual HTML for disabled checkbox:
<input id='section-966c9ee8-d471-48a9-afe3-3246e43633d8' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-966c9ee8-d471-48a9-afe3-3246e43633d8' class='xr-section-summary'  title='Expand/collapse section'>Dims: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'></div>
```
</details>

## Why This Is A Bug

The bug violates standard UX principles where tooltips should guide user interaction on interactive elements, not on disabled elements. The root cause is a variable type confusion in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/core/formatting_html.py`:

1. **Line 181** reassigns the boolean parameter `enabled` to a string value:
   - `enabled = "" if enabled and has_items else "disabled"`
   - After this, `enabled` is either `""` (empty string) or `"disabled"` (non-empty string)

2. **Line 183** attempts to use `enabled` as a boolean to determine tooltip display:
   - `tip = " title='Expand/collapse section'" if enabled else ""`
   - Empty string `""` evaluates to `False` in Python (no tooltip for enabled checkboxes)
   - Non-empty string `"disabled"` evaluates to `True` in Python (tooltip shown for disabled checkboxes)

This creates backwards behavior where disabled checkboxes that cannot be interacted with display an "Expand/collapse section" tooltip, while enabled checkboxes that users can actually click have no tooltip guidance.

## Relevant Context

- The function is used internally by xarray for HTML representation in Jupyter notebooks
- The bug affects real usage: `dim_section()` at line 230 calls `collapsible_section` with `enabled=False`, producing disabled checkboxes with misleading tooltips
- While the function is not part of the public API, it impacts the user experience when viewing xarray objects in Jupyter environments
- The collapsible sections still function correctly; only the tooltip hints are inverted
- Documentation: No formal documentation exists for this internal function

## Proposed Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -177,11 +177,12 @@ def collapsible_section(
     data_id = "section-" + str(uuid.uuid4())

     has_items = n_items is not None and n_items
     n_items_span = "" if n_items is None else f" <span>({n_items})</span>"
-    enabled = "" if enabled and has_items else "disabled"
+    is_enabled = enabled and has_items
+    enabled_attr = "" if is_enabled else "disabled"
     collapsed = "" if collapsed or not has_items else "checked"
-    tip = " title='Expand/collapse section'" if enabled else ""
+    tip = " title='Expand/collapse section'" if is_enabled else ""

     return (
         f"<input id='{data_id}' class='xr-section-summary-in' "
-        f"type='checkbox' {enabled} {collapsed}>"
+        f"type='checkbox' {enabled_attr} {collapsed}>"
         f"<label for='{data_id}' class='xr-section-summary' {tip}>"
```