# Bug Report: xarray.static.css Missing Comma in CSS var() Function

**Target**: `xarray.static.css`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The CSS file at `xarray/static/css/style.css` contains a syntax error on line 8 where a required comma is missing between the CSS custom property name and its fallback value in a `var()` function call.

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
            # Look for var() calls that have rgba() as fallback but missing comma
            var_match = re.search(r'var\(--[a-zA-Z0-9-]+\s+rgba\(', line)
            if var_match:
                invalid_lines.append((line_num, line.strip()))

    assert not invalid_lines, f"Found CSS var() calls with missing comma before fallback value:\n" + \
        "\n".join([f"  Line {num}: {line}" for num, line in invalid_lines])


if __name__ == "__main__":
    try:
        test_css_var_function_syntax()
        print("Test passed: No CSS syntax errors found")
    except AssertionError as e:
        print(f"Test failed: {e}")
```

<details>

<summary>
**Failing input**: Line 8 of `xarray/static/css/style.css`
</summary>
```
Test failed: Found CSS var() calls with missing comma before fallback value:
  Line 8: var(--pst-color-text-base rgba(0, 0, 0, 1))
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files

# Load the CSS content
icons_svg, css_content = _load_static_files()

# Split the CSS into lines
lines = css_content.split('\n')

# Show the problematic line (line 8, 0-indexed as line 7)
print(f"Line 8 of xarray/static/css/style.css:")
print(f"  {lines[7]}")

# Check for the syntax error
if 'var(--pst-color-text-base rgba(' in lines[7]:
    print("\nERROR: Missing comma in CSS var() function!")
    print("  Found: var(--pst-color-text-base rgba(0, 0, 0, 1))")
    print("  Expected: var(--pst-color-text-base, rgba(0, 0, 0, 1))")
    print("\nThis violates CSS Custom Properties specification which requires:")
    print("  var( <custom-property-name> , <declaration-value>? )")
    print("  The comma between property name and fallback value is REQUIRED.")

# Show correct usage in the same file for comparison
print("\nCorrect usage of the same variable elsewhere in the file:")
for i, line in enumerate(lines[:60], 1):
    if 'var(--pst-color-text-base,' in line:
        print(f"  Line {i}: {line.strip()}")
```

<details>

<summary>
CSS syntax error detected on line 8
</summary>
```
Line 8 of xarray/static/css/style.css:
      var(--pst-color-text-base rgba(0, 0, 0, 1))

ERROR: Missing comma in CSS var() function!
  Found: var(--pst-color-text-base rgba(0, 0, 0, 1))
  Expected: var(--pst-color-text-base, rgba(0, 0, 0, 1))

This violates CSS Custom Properties specification which requires:
  var( <custom-property-name> , <declaration-value>? )
  The comma between property name and fallback value is REQUIRED.

Correct usage of the same variable elsewhere in the file:
  Line 12: var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
  Line 16: var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
  Line 46: var(--pst-color-text-base, rgba(255, 255, 255, 1))
  Line 50: var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
  Line 54: var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
```
</details>

## Why This Is A Bug

The CSS Custom Properties specification (W3C CSS Custom Properties for Cascading Variables Module Level 1) explicitly defines the `var()` function syntax as:

```
var( <custom-property-name> , <declaration-value>? )
```

The comma between the custom property name and the optional fallback value is **mandatory** when a fallback is provided. Line 8 violates this requirement by omitting the comma:

- **Incorrect (line 8)**: `var(--pst-color-text-base rgba(0, 0, 0, 1))`
- **Correct (lines 12, 16, 46, 50, 54)**: `var(--pst-color-text-base, rgba(0, 0, 0, 1))`

Without the comma, CSS parsers cannot distinguish where the custom property name ends and the fallback value begins. This causes:
1. The entire expression `--pst-color-text-base rgba(0, 0, 0, 1)` to be interpreted as a malformed custom property name
2. The fallback value `rgba(0, 0, 0, 1)` will not be applied when `--pst-color-text-base` is undefined
3. Potential rendering inconsistencies across different browsers and CSS validators

## Relevant Context

This CSS file is used for styling xarray objects when displayed in JupyterLab environments. The bug affects the `--xr-font-color0` CSS custom property which controls text color in xarray's HTML representations.

The same variable `--pst-color-text-base` is used correctly with commas in 5 other locations within the same file, making this clearly an accidental typo on line 8. The pattern shows this is for PyData Sphinx Theme integration, providing fallback colors when theme variables aren't defined.

CSS specification reference: https://www.w3.org/TR/css-variables-1/#using-variables

The file is located at: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/static/css/style.css`

## Proposed Fix

```diff
--- a/xarray/static/css/style.css
+++ b/xarray/static/css/style.css
@@ -5,7 +5,7 @@
 :root {
   --xr-font-color0: var(
     --jp-content-font-color0,
-    var(--pst-color-text-base rgba(0, 0, 0, 1))
+    var(--pst-color-text-base, rgba(0, 0, 0, 1))
   );
   --xr-font-color2: var(
     --jp-content-font-color2,
```