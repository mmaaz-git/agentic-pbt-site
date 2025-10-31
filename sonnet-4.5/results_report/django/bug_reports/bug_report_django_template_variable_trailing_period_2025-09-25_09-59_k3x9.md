# Bug Report: django.template.Variable Trailing Period

**Target**: `django.template.Variable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a Variable is created with a number followed by a trailing period (e.g., "10."), both the `literal` and `lookups` attributes are set, creating an inconsistent internal state. Additionally, calling `resolve()` on such a variable raises `VariableDoesNotExist` instead of returning the literal value.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import django.template

@given(st.integers(min_value=0, max_value=1000))
def test_integer_trailing_period_property(num):
    text = f"{num}."
    var = django.template.Variable(text)

    if var.literal is not None and var.lookups is not None:
        assert False, f"Both literal ({var.literal}) and lookups ({var.lookups}) are set for '{text}'"
```

**Failing input**: `num=0` (or any integer)

## Reproducing the Bug

```python
from django.template import Variable

var = Variable("10.")

print(f"literal: {var.literal}")
print(f"lookups: {var.lookups}")

assert var.literal == 10.0
assert var.lookups == ('10', '')

try:
    result = var.resolve({})
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

**Output:**
```
literal: 10.0
lookups: ('10', '')
VariableDoesNotExist: Failed lookup for key [10] in {}
```

## Why This Is A Bug

1. **Inconsistent internal state**: The Variable object has both `literal` and `lookups` set, which should be mutually exclusive. Either it's a literal value or it's a variable lookup, not both.

2. **Incorrect behavior**: When `literal` is set to a valid float (10.0), `resolve()` should return that literal value. Instead, it raises `VariableDoesNotExist` because `lookups` is also set.

3. **Violates documented behavior**: The code comment in the Variable `__init__` method states `# "2." is invalid` and includes logic to raise ValueError for this case. However, the ValueError is only raised when the string ends with a period AND contains 'e' or '.', which means "2." passes the float parsing but then gets treated as a variable lookup.

4. **Violates principle of least surprise**: A template author using `{{ 10. }}` would reasonably expect it to either:
   - Be treated as the float 10.0 (like Python does), or
   - Raise a clear TemplateSyntaxError saying it's invalid syntax

   Instead, it's treated as a variable lookup for a key named "10" with an empty attribute "", which is confusing.

## Fix

The issue is in `django/template/base.py` in the `Variable.__init__` method. After successfully parsing a float, the code continues to process the variable as if it might be a lookup. The fix should ensure that when `literal` is set, the method returns early without setting `lookups`.

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -825,7 +825,11 @@ class Variable:
             if "." in var or "e" in var.lower():
                 self.literal = float(var)
                 # "2." is invalid
                 if var[-1] == ".":
-                    raise ValueError
+                    # Explicitly reject trailing periods in numeric literals
+                    # to avoid ambiguity with attribute lookup syntax
+                    raise ValueError(
+                        f"Numeric literals cannot end with a period: {var!r}"
+                    )
             else:
                 self.literal = int(var)
+            return  # Early return when literal is set successfully
         except ValueError:
```

Alternatively, if the intention is to support trailing periods in floats (which Python's `float()` function does), then the code should not set `lookups` when `literal` is already set:

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -840,6 +840,9 @@ class Variable:
         except ValueError:
+            # Only process as variable lookup if literal was not set
+            if self.literal is not None:
+                return
+
             # Otherwise we'll set self.lookups so that resolve() knows we're
             # dealing with a bonafide variable
```