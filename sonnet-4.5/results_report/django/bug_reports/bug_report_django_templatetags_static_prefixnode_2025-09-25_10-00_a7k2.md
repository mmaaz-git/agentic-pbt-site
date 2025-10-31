# Bug Report: django.templatetags.static PrefixNode IndexError

**Target**: `django.templatetags.static.PrefixNode.handle_token`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `PrefixNode.handle_token` method raises an `IndexError` instead of a proper `TemplateSyntaxError` when parsing malformed template tags that have 'as' without a following variable name.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.template import Template, Context, TemplateSyntaxError

@given(st.text(min_size=1, max_size=50))
def test_prefix_node_malformed_as_clause(text):
    assume('"' not in text and "'" not in text)
    assume(' ' not in text and text != 'as')

    template_str = "{% load static %}{% get_static_prefix as %}"

    try:
        template = Template(template_str)
    except TemplateSyntaxError:
        pass
    except IndexError:
        assert False, "Should raise TemplateSyntaxError, not IndexError"
```

**Failing input**: Template string `{% get_static_prefix as %}`

## Reproducing the Bug

```python
from django.template import Template, Context, TemplateSyntaxError

template_str = "{% load static %}{% get_static_prefix as %}"

try:
    template = Template(template_str)
    print("Template compiled successfully")
except TemplateSyntaxError as e:
    print(f"Correctly raised TemplateSyntaxError: {e}")
except IndexError as e:
    print(f"BUG: IndexError raised: {e}")
```

## Why This Is A Bug

When a user writes a malformed template tag like `{% get_static_prefix as %}` (missing the variable name after 'as'), Django should raise a clear `TemplateSyntaxError` explaining the problem. Instead, it crashes with an `IndexError` because the code tries to access `tokens[2]` when only `tokens[0]` and `tokens[1]` exist.

In `static.py` lines 30-38:
- Line 31 checks if `tokens[1]` exists and equals "as"
- Line 35 checks if there are more than 1 token
- Line 36 blindly accesses `tokens[2]` without checking if it exists

The bug occurs when `tokens = ['get_static_prefix', 'as']` (2 tokens):
- Line 35: `len(tokens) > 1` is True (2 > 1)
- Line 36: `tokens[2]` raises IndexError (only indices 0 and 1 exist)

## Fix

```diff
--- a/django/templatetags/static.py
+++ b/django/templatetags/static.py
@@ -32,8 +32,13 @@ class PrefixNode(template.Node):
             raise template.TemplateSyntaxError(
                 "First argument in '%s' must be 'as'" % tokens[0]
             )
-        if len(tokens) > 1:
+        if len(tokens) == 3:
             varname = tokens[2]
+        elif len(tokens) == 1:
+            varname = None
         else:
-            varname = None
+            raise template.TemplateSyntaxError(
+                "'%s' requires 'as variable' or no arguments (got %r)"
+                % (tokens[0], tokens[1:])
+            )
         return cls(varname, name)
```