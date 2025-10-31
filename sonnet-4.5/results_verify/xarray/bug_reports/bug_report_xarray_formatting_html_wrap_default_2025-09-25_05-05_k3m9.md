# Bug Report: xarray.core.formatting_html._wrap_datatree_repr Default Parameter Documentation

**Target**: `xarray.core.formatting_html._wrap_datatree_repr`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_wrap_datatree_repr` function's docstring incorrectly states that the default value for the `end` parameter is `True`, when the actual default value in the function signature is `False`. This creates a discrepancy between documented and actual behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import xarray.core.formatting_html as fmt_html

@given(st.text(min_size=1))
def test_wrap_default_matches_documented_behavior(html_input):
    """
    Test that the default behavior of _wrap_datatree_repr matches its documentation.

    The docstring says: "Default is True."
    Therefore, calling the function without the end parameter should behave
    the same as calling it with end=True.
    """
    result_default = fmt_html._wrap_datatree_repr(html_input)
    result_true = fmt_html._wrap_datatree_repr(html_input, end=True)

    # According to the docstring, these should be the same
    assert result_default == result_true, \
        "Default behavior should match end=True as documented"
```

**Failing input**: Any text input (e.g., `"<div>test</div>"`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import xarray.core.formatting_html as fmt_html
import inspect

test_html = "<div>Test Content</div>"

result_default = fmt_html._wrap_datatree_repr(test_html)
result_false = fmt_html._wrap_datatree_repr(test_html, end=False)
result_true = fmt_html._wrap_datatree_repr(test_html, end=True)

sig = inspect.signature(fmt_html._wrap_datatree_repr)
actual_default = sig.parameters['end'].default

print(f"Actual default value in signature: {actual_default}")
print(f"Documented default (from docstring): True")
print()
print(f"Default behavior matches end=False: {result_default == result_false}")
print(f"Default behavior matches end=True:  {result_default == result_true}")
```

Output:
```
Actual default value in signature: False
Documented default (from docstring): True

Default behavior matches end=False: True
Default behavior matches end=True:  False
```

## Why This Is A Bug

The function's docstring (line 465 in `formatting_html.py`) explicitly states "Default is True", but the function signature (line 437) defines `end: bool = False`. This is a contract violation where the documentation contradicts the implementation. Users relying on the documentation will have incorrect expectations about the function's behavior.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -462,7 +462,7 @@ def _wrap_datatree_repr(r: str, end: bool = False) -> str:
     end: bool
         Specify if the line on the left should continue or end.

-        Default is True.
+        Default is False.

     Returns
     -------
```