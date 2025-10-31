# Bug Report: xarray.plot _nicetitle maxchar violation

**Target**: `xarray.plot.facetgrid._nicetitle`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_nicetitle` function violates its documented behavior when `maxchar < 3`. The function is supposed to truncate titles to at most `maxchar` characters, but when the title needs truncation and `maxchar < 3`, it returns a string longer than `maxchar`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.plot.facetgrid import _nicetitle


@given(
    st.text(min_size=1, max_size=10),
    st.text(min_size=1, max_size=10),
    st.integers(min_value=1, max_value=3),
)
def test_nicetitle_small_maxchar(coord, value, maxchar):
    template = "{coord}={value}"
    result = _nicetitle(coord, value, maxchar, template)

    assert len(result) <= maxchar, \
        f"Result length {len(result)} exceeds maxchar {maxchar}: '{result}'"
```

**Failing input**: `coord='0'`, `value='0'`, `maxchar=1`

## Reproducing the Bug

```python
from xarray.plot.facetgrid import _nicetitle

result = _nicetitle("x", "1", maxchar=1, template="{coord}={value}")
print(f"Result: '{result}'")
print(f"Length: {len(result)}")
print(f"Expected max length: 1")
```

Output:
```
Result: 'x...'
Length: 4
Expected max length: 1
```

The function returns `'x...'` (4 characters) when `maxchar=1`, violating the constraint that the result should be at most 1 character.

Additional examples:
- `maxchar=0`: returns `'...'` (3 characters) instead of max 0
- `maxchar=2`: returns `'x=...'` (5 characters) instead of max 2

## Why This Is A Bug

The function's docstring states: "Put coord, value in template and truncate at maxchar". The implementation adds `"..."` (3 characters) to indicate truncation without checking if this makes the result longer than `maxchar`. When `maxchar < 3`, the ellipsis itself exceeds the limit.

From `facetgrid.py:46-56`:
```python
def _nicetitle(coord, value, maxchar, template):
    """
    Put coord, value in template and truncate at maxchar
    """
    prettyvalue = format_item(value, quote_strings=False)
    title = template.format(coord=coord, value=prettyvalue)

    if len(title) > maxchar:
        title = title[: (maxchar - 3)] + "..."  # BUG: doesn't ensure len <= maxchar

    return title
```

When `maxchar < 3`, `title[:(maxchar - 3)]` becomes negative indexing (e.g., `title[:-2]`), which still returns characters, and then `"..."` is appended, resulting in a string longer than `maxchar`.

## Fix

```diff
def _nicetitle(coord, value, maxchar, template):
    """
    Put coord, value in template and truncate at maxchar
    """
    prettyvalue = format_item(value, quote_strings=False)
    title = template.format(coord=coord, value=prettyvalue)

    if len(title) > maxchar:
-       title = title[: (maxchar - 3)] + "..."
+       if maxchar >= 3:
+           title = title[: (maxchar - 3)] + "..."
+       else:
+           title = title[:maxchar]

    return title
```

This ensures that:
- When `maxchar >= 3`: Use ellipsis to indicate truncation
- When `maxchar < 3`: Simply truncate to `maxchar` without ellipsis (no room for it)