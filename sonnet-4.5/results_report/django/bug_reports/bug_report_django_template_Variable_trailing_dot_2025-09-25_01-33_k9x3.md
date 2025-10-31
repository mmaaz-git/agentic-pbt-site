# Bug Report: django.template.Variable Trailing Dot Inconsistency

**Target**: `django.template.Variable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a Variable is initialized with a numeric string ending in a dot (e.g., `'42.'`), the internal state becomes inconsistent with both `literal` and `lookups` attributes set, causing resolution to fail even though the Variable was successfully created.

## Property-Based Test

```python
from django.template import Variable, Context
from hypothesis import given, settings, strategies as st, example


@settings(max_examples=100)
@example(n=42)
@example(n=0)
@example(n=123)
@given(st.integers())
def test_variable_numeric_string_with_trailing_dot_should_be_resolvable(n):
    s = str(n) + '.'
    var = Variable(s)

    if var.literal is not None:
        ctx = Context({})
        resolved = var.resolve(ctx)
        assert resolved == var.literal
```

**Failing input**: `n=42` (and any integer)

## Reproducing the Bug

```python
from django.template import Variable, Context

var = Variable('42.')
print(f"literal: {var.literal}")
print(f"lookups: {var.lookups}")

ctx = Context({})
result = var.resolve(ctx)
```

Output:
```
literal: 42.0
lookups: ('42', '')
VariableDoesNotExist: Failed lookup for key [42] in [...]
```

## Why This Is A Bug

The Variable class should maintain an invariant: either `literal` or `lookups` should be set, never both. When `'42.'` is parsed:

1. `float('42.')` succeeds and sets `literal = 42.0`
2. The check `if var[-1] == ".": raise ValueError` intentionally raises ValueError
3. The ValueError handler then sets `lookups = ('42', '')`
4. Now both `literal` and `lookups` are set

The `resolve()` method checks `if self.lookups is not None` first, so it ignores the literal value and tries to resolve via lookups, which fails because the empty string lookup component is invalid.

This violates the expected behavior: strings like `'42.'` should either (a) be accepted as valid floats, or (b) be rejected during initialization. The current behavior accepts them during initialization but causes failures during resolution.

## Fix

The issue is that when ValueError is raised after successfully setting `self.literal`, the exception handler unconditionally sets `self.lookups` without clearing `self.literal`. The fix is to clear `self.literal` when falling back to variable lookup:

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -832,6 +832,7 @@ class Variable:
         except ValueError:
             # A ValueError means that the variable isn't a number.
+            self.literal = None
             if var[0:2] == "_(" and var[-1] == ")":
                 # The result of the lookup should be translated at rendering
                 # time.
```

This ensures that `literal` and `lookups` are mutually exclusive, maintaining the class invariant.