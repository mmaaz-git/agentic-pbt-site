# Bug Report: django.template.Variable Trailing Dot Parsing

**Target**: `django.template.Variable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`Variable` incorrectly accepts numeric literals with trailing dots (e.g., "2.", "0.") despite a code comment indicating they should be invalid. This creates inconsistent internal state where both `literal` and `lookups` are set, causing the variable to resolve via lookup instead of using the literal value.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.template import Variable

@given(
    st.integers(),
    st.booleans()
)
def test_variable_literal_trailing_dot_bug(num, add_trailing_dot):
    var_string = str(num)
    if add_trailing_dot:
        var_string = var_string + '.'

    try:
        v = Variable(var_string)

        if v.literal is not None and isinstance(v.literal, float):
            assert var_string[-1] != '.', f"Variable with trailing dot '{var_string}' should not parse as valid float but got {v.literal}"
    except ValueError:
        pass
```

**Failing input**: `num=0, add_trailing_dot=True` (produces "0.")

## Reproducing the Bug

```python
from django.template import Variable, Context

v = Variable("2.")

print(f"literal = {v.literal}")
print(f"lookups = {v.lookups}")

ctx = Context({"2": {"": "unexpected_value"}})
result = v.resolve(ctx)
print(f"Result: {result}")
```

Output:
```
literal = 2.0
lookups = ('2', '')
Result: unexpected_value
```

## Why This Is A Bug

The Variable.__init__ code contains a comment stating "2. is invalid" and includes a check to raise ValueError for trailing dots. However, the check occurs AFTER `self.literal = float(var)`, which means:

1. The literal gets set to the parsed float value
2. The ValueError is raised (correctly)
3. The exception handler then treats the string as a variable name
4. Both `literal` and `lookups` end up set, violating the intended invariant

When resolving, `lookups` takes precedence over `literal`, so "2." resolves as a variable lookup instead of the literal 2.0. This is inconsistent with the intended behavior and the code comment.

## Fix

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -559,10 +559,10 @@ class Variable:
             # Try to interpret values containing a period or an 'e'/'E'
             # (possibly scientific notation) as a float;  otherwise, try int.
             if "." in var or "e" in var.lower():
-                self.literal = float(var)
                 # "2." is invalid
                 if var[-1] == ".":
                     raise ValueError
+                self.literal = float(var)
             else:
                 self.literal = int(var)
         except ValueError:
```

The fix moves the trailing dot check before the `float()` call, ensuring that invalid strings don't set `literal` before raising ValueError.