# Bug Report: xarray.static.css CSS var() Syntax Error

**Target**: `xarray.static.css`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The CSS file at `xarray/static/css/style.css` contains a syntax error in a CSS `var()` function call on line 8, where the required comma between the variable name and fallback value is missing.

## Property-Based Test

```python
import sys
import re

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files


def test_css_var_function_syntax():
    icons_svg, css_content = _load_static_files()

    lines = css_content.split('\n')
    invalid_lines = []

    for line_num, line in enumerate(lines, 1):
        if 'var(' in line and 'rgba(' in line:
            var_match = re.search(r'var\(--[a-zA-Z0-9-]+\s+rgba\(', line)
            if var_match:
                invalid_lines.append((line_num, line.strip()))

    assert not invalid_lines, f"Found CSS var() calls with missing comma before fallback value:\n" + \
        "\n".join([f"  Line {num}: {line}" for num, line in invalid_lines])
```

**Failing input**: Line 8 of `xarray/static/css/style.css`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files

icons_svg, css_content = _load_static_files()
lines = css_content.split('\n')

print("Line 8:", lines[7])
```

Output:
```
Line 8:     var(--pst-color-text-base rgba(0, 0, 0, 1))
```

## Why This Is A Bug

According to the [CSS Custom Properties specification](https://www.w3.org/TR/css-variables-1/), the `var()` function syntax is:

```
var( <custom-property-name> , <declaration-value>? )
```

When providing a fallback value (the second argument), a comma is **required** to separate it from the custom property name. The current code is missing this comma on line 8:

**Incorrect**: `var(--pst-color-text-base rgba(0, 0, 0, 1))`
**Correct**: `var(--pst-color-text-base, rgba(0, 0, 0, 1))`

This syntax error means:
1. The CSS may not parse correctly in all browsers
2. The fallback value `rgba(0, 0, 0, 1)` will not be applied when `--pst-color-text-base` is undefined
3. This violates CSS standards and could cause styling issues

The same variable is used correctly with a comma on lines 12, 16, 46, 50, and 54, making line 8 an obvious typo.

## Fix

Add the missing comma on line 8 of `xarray/static/css/style.css`:

```diff
 :root {
   --xr-font-color0: var(
     --jp-content-font-color0,
-    var(--pst-color-text-base rgba(0, 0, 0, 1))
+    var(--pst-color-text-base, rgba(0, 0, 0, 1))
   );
   --xr-font-color2: var(
     --jp-content-font-color2,
     var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
   );
```