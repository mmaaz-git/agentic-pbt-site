# Bug Report: django.templatetags.static.StaticNode.handle_token IndexError

**Target**: `django.templatetags.static.StaticNode.handle_token`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `StaticNode.handle_token` method raises `IndexError` when parsing template tokens with 3 elements where the second-to-last element is "as", due to incorrect bounds checking before accessing `bits[3]`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.template import TemplateSyntaxError
from django.templatetags.static import StaticNode


class MockToken:
    def __init__(self, contents):
        self.contents = contents

    def split_contents(self):
        return self.contents.split()


class MockParser:
    def compile_filter(self, token):
        return token


@given(st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=10))
@settings(max_examples=1000)
def test_static_node_handle_token_no_index_error(token_parts):
    token = MockToken(' '.join(token_parts))
    parser = MockParser()

    try:
        result = StaticNode.handle_token(parser, token)
    except (TemplateSyntaxError, AttributeError):
        pass
    except IndexError as e:
        raise AssertionError(f"IndexError with token parts: {token_parts}")
```

**Failing input**: `['x', 'as', 'y']`

## Reproducing the Bug

```python
from django.templatetags.static import StaticNode


class MockToken:
    def __init__(self, contents):
        self.contents = contents

    def split_contents(self):
        return self.contents.split()


class MockParser:
    def compile_filter(self, token):
        return token


parser = MockParser()
token = MockToken('x as y')

StaticNode.handle_token(parser, token)
```

## Why This Is A Bug

The code at lines 147-148 in `static.py` checks if `len(bits) >= 2 and bits[-2] == "as"`, but then unconditionally accesses `bits[3]`. When `bits` has exactly 3 elements like `['x', 'as', 'y']`, the condition is satisfied (since `bits[-2] == bits[1] == 'as'`), but `bits[3]` doesn't exist, causing an `IndexError`.

This violates the expected behavior where the method should either successfully parse valid template syntax or raise a `TemplateSyntaxError` for invalid syntax, not crash with an `IndexError`.

## Fix

```diff
--- a/django/templatetags/static.py
+++ b/django/templatetags/static.py
@@ -144,7 +144,7 @@ class StaticNode(template.Node):

         path = parser.compile_filter(bits[1])

-        if len(bits) >= 2 and bits[-2] == "as":
+        if len(bits) >= 4 and bits[-2] == "as":
             varname = bits[3]
         else:
             varname = None
```

The fix changes the condition from `len(bits) >= 2` to `len(bits) >= 4`, ensuring that `bits[3]` is accessible before attempting to access it.