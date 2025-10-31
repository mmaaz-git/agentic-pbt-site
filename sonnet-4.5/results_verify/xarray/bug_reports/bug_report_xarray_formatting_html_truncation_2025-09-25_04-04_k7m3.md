# Bug Report: xarray.core.formatting_html.summarize_datatree_children Truncation with max_children=1

**Target**: `xarray.core.formatting_html.summarize_datatree_children`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `display_max_children=1` and there are multiple children to display, `summarize_datatree_children` only shows the first child without showing any children from the end. This is inconsistent with the behavior for larger `max_children` values, which show children from both the beginning and end of the list.

## Property-Based Test

```python
import xarray as xr
from xarray.core.datatree import DataTree
from xarray.core.options import set_options
from xarray.core.formatting_html import summarize_datatree_children
from hypothesis import given, strategies as st, assume

@given(
    n_children=st.integers(min_value=2, max_value=20),
    max_children=st.integers(min_value=1, max_value=10)
)
def test_truncation_shows_both_ends_when_truncating(n_children, max_children):
    """
    Property: When n_children > max_children, the output should show
    children from BOTH the beginning and the end of the list.
    """
    assume(n_children > max_children)

    children = {}
    for i in range(n_children):
        ds = xr.Dataset({f"var{i}": ([f"dim{i}"], [1, 2, 3])})
        children[f"child{i}"] = DataTree(ds)

    with set_options(display_max_children=max_children):
        result = summarize_datatree_children(children)

    first_child_shown = "child0" in result
    last_child_shown = f"child{n_children-1}" in result

    assert first_child_shown, "First child should always be shown when truncating"
    assert last_child_shown, f"Last child should be shown when truncating"
```

**Failing input**: `max_children=1, n_children=2` (minimal example found by Hypothesis)

## Reproducing the Bug

```python
import xarray as xr
from xarray.core.datatree import DataTree
from xarray.core.options import set_options
from xarray.core.formatting_html import summarize_datatree_children

children = {
    "child0": DataTree(xr.Dataset({"var0": (["x"], [1, 2, 3])})),
    "child1": DataTree(xr.Dataset({"var1": (["y"], [4, 5, 6])})),
}

with set_options(display_max_children=1):
    result = summarize_datatree_children(children)

    print(f"'child0' in result: {'child0' in result}")
    print(f"'child1' in result: {'child1' in result}")
```

**Output:**
```
'child0' in result: True
'child1' in result: False
```

## Why This Is A Bug

The truncation logic in `summarize_datatree_children` (lines 364-384) calculates which children to show using:

```python
if i < ceil(MAX_CHILDREN / 2) or i >= ceil(n_children - MAX_CHILDREN / 2):
    # show this child
```

When `MAX_CHILDREN=1` and `n_children=2`:
- First half: show indices `i < ceil(1/2) = 1` → shows index 0 ✓
- Last half: show indices `i >= ceil(2 - 0.5) = 2` → shows nothing (max index is 1) ✗

This means no children from the end are displayed, which is inconsistent with the behavior for larger `max_children` values (e.g., when `max_children=2`, both first and last children are shown).

Users who set `display_max_children=1` (e.g., to minimize output) would reasonably expect to see at least one representative child, but the ellipsis "..." suggests there are more children that aren't shown - yet none from the end are displayed.

## Fix

The issue is that the calculation for `last_half_start` can exceed the valid index range when `MAX_CHILDREN` is very small. The fix should ensure that when truncating, we always show at least one element from the end:

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -367,8 +367,11 @@ def summarize_datatree_children(children: Mapping[str, DataTree]) -> str:

     children_html = []
     for i, (n, c) in enumerate(children.items()):
-        if i < ceil(MAX_CHILDREN / 2) or i >= ceil(n_children - MAX_CHILDREN / 2):
+        first_half_end = ceil(MAX_CHILDREN / 2)
+        last_half_start = max(first_half_end, n_children - max(1, MAX_CHILDREN - first_half_end))
+
+        if i < first_half_end or i >= last_half_start:
             is_last = i == (n_children - 1)
             children_html.append(
                 _wrap_datatree_repr(datatree_node_repr(n, c), end=is_last)
             )
-        elif n_children > MAX_CHILDREN and i == ceil(MAX_CHILDREN / 2):
+        elif n_children > MAX_CHILDREN and i == first_half_end:
             children_html.append("<div>...</div>")