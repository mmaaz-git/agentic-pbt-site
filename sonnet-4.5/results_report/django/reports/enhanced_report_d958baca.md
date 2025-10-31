# Bug Report: django.templatetags.static.StaticNode.handle_token IndexError on Malformed Template Syntax

**Target**: `django.templatetags.static.StaticNode.handle_token`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `StaticNode.handle_token` method raises an `IndexError` instead of the expected `TemplateSyntaxError` when parsing malformed template tokens with exactly 3 elements where the second element is "as".

## Property-Based Test

```python
#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

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


if __name__ == "__main__":
    test_static_node_handle_token_no_index_error()
```

<details>

<summary>
**Failing input**: `['x', 'as', 'y']`
</summary>
```
Traceback (most recent call last):
  File "<string>", line 27, in test_static_node_handle_token_no_index_error
    result = StaticNode.handle_token(parser, token)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/templatetags/static.py", line 148, in handle_token
    varname = bits[3]
              ~~~~^^^
IndexError: list index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 33, in <module>
    test_static_node_handle_token_no_index_error()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "<string>", line 20, in test_static_node_handle_token_no_index_error
    @example(['x', 'as', 'y'])
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "<string>", line 31, in test_static_node_handle_token_no_index_error
    raise AssertionError(f'IndexError with token parts: {token_parts}')
AssertionError: IndexError with token parts: ['x', 'as', 'y']
Falsifying explicit example: test_static_node_handle_token_no_index_error(
    token_parts=['x', 'as', 'y'],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.templatetags.static import StaticNode


class MockToken:
    def __init__(self, contents):
        self.contents = contents

    def split_contents(self):
        return self.contents.split()


class MockParser:
    def compile_filter(self, token):
        return token


# Create test case that triggers the IndexError
parser = MockParser()
token = MockToken('x as y')

print("Testing StaticNode.handle_token with token: 'x as y'")
print(f"Token split_contents() returns: {token.split_contents()}")
print()

try:
    result = StaticNode.handle_token(parser, token)
    print(f"Success: returned {result}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other exception raised: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
IndexError: list index out of range
</summary>
```
Testing StaticNode.handle_token with token: 'x as y'
Token split_contents() returns: ['x', 'as', 'y']

IndexError raised: list index out of range
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/repo.py", line 32, in <module>
    result = StaticNode.handle_token(parser, token)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/templatetags/static.py", line 148, in handle_token
    varname = bits[3]
              ~~~~^^^
IndexError: list index out of range
```
</details>

## Why This Is A Bug

This violates Django's documented template tag error handling conventions. According to Django's documentation, template tags must:

1. **Validate their syntax properly** - Template tags are responsible for validating input arguments
2. **Raise TemplateSyntaxError for invalid syntax** - The documentation explicitly states: "This function is responsible for raising django.template.TemplateSyntaxError, with helpful messages, for any syntax error"
3. **Never expose implementation exceptions** - Python exceptions like IndexError should not bubble up to template authors

The bug occurs because the code at lines 147-148 in `django/templatetags/static.py` checks `if len(bits) >= 2 and bits[-2] == "as"` but then unconditionally accesses `bits[3]`. When `bits` has exactly 3 elements (e.g., `['x', 'as', 'y']`), the condition evaluates to True because:
- `len(bits) = 3` which is `>= 2`: True
- `bits[-2] = bits[1] = 'as'`: True

However, `bits[3]` doesn't exist in a 3-element list (valid indices are 0, 1, 2), causing the IndexError.

## Relevant Context

The `{% static %}` template tag is documented to accept the following valid syntax forms:
- `{% static "path/to/file.css" %}` - Basic usage
- `{% static variable_with_path %}` - With variable
- `{% static "path/to/file.css" as varname %}` - Store in variable
- `{% static variable_with_path as varname %}` - Variable stored in variable

The failing input `['x', 'as', 'y']` would represent the malformed template syntax `x as y` (missing the tag name "static" at the beginning). While this is clearly invalid syntax, Django's convention is to handle such cases gracefully with a `TemplateSyntaxError`.

Documentation references:
- Django template tag development: https://docs.djangoproject.com/en/stable/howto/custom-template-tags/
- Static files handling: https://docs.djangoproject.com/en/stable/ref/templates/builtins/#static

Code location: `/django/templatetags/static.py` lines 147-148

## Proposed Fix

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