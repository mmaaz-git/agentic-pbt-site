# Bug Report: django.template.Variable Trailing Dot Handling

**Target**: `django.template.Variable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Variable.__init__() attempts to reject numeric strings with trailing dots (e.g., "2.") but incorrectly treats them as valid variable lookups due to improper exception handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
from django.template import Variable

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1, max_value=1e10))
def test_variable_float_with_trailing_dot_should_be_rejected(num):
    """
    Property: Floats with trailing dots should be rejected as invalid.
    Evidence: Code comment on line 824 says '"2." is invalid' and code
    explicitly raises ValueError for this case on line 826.
    """
    var_str = f"{int(num)}."

    with pytest.raises((ValueError, Exception)):
        var = Variable(var_str)
```

**Failing input**: `"2."`, `"1."`, `"10."`, or any integer followed by a dot

## Reproducing the Bug

```python
from django.template import Variable

var = Variable("2.")

print(f"literal: {var.literal}")
print(f"lookups: {var.lookups}")
```

**Output:**
```
literal: 2.0
lookups: ('2', '')
```

## Why This Is A Bug

1. The code explicitly intends to reject strings like "2." as invalid (see comment on line 824: `# "2." is invalid`)
2. The code raises ValueError on line 826 to reject this case
3. However, the ValueError is caught by the outer `except ValueError` block starting on line 829
4. The string "2." is then treated as a variable name and split on `.` creating `lookups = ('2', '')`
5. This creates a malformed Variable object with an empty string as a lookup component
6. This violates the documented intention and creates unexpected behavior

The bug affects any numeric string with a trailing dot, causing them to be treated as dotted variable lookups with an empty trailing component instead of being rejected as invalid.

## Fix

The issue is that the ValueError raised to reject trailing dot numbers is caught by the outer except block. The fix is to check for the trailing dot BEFORE attempting the float conversion, or use a specific exception type:

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -820,10 +820,10 @@ class Variable:
             # Try to interpret values containing a period or an 'e'/'E'
             # (possibly scientific notation) as a float;  otherwise, try int.
             if "." in var or "e" in var.lower():
-                self.literal = float(var)
                 # "2." is invalid
                 if var[-1] == ".":
-                    raise ValueError
+                    raise TemplateSyntaxError(f"Invalid numeric literal: '{var}'")
+                self.literal = float(var)
             else:
                 self.literal = int(var)
         except ValueError: