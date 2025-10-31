# Bug Report: django.templatetags.static.PrefixNode Raises IndexError for Malformed Template Tags

**Target**: `django.templatetags.static.PrefixNode.handle_token`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `PrefixNode.handle_token` method crashes with an `IndexError` instead of raising the expected `TemplateSyntaxError` when parsing template tags that contain 'as' without a following variable name.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for django.templatetags.static.PrefixNode bug
"""

import os
import sys
import django
from django.conf import settings

# Configure minimal Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': False,
            'OPTIONS': {
                'context_processors': [],
            },
        }],
        INSTALLED_APPS=[
            'django.contrib.staticfiles',
        ],
        STATIC_URL='/static/',
        MEDIA_URL='/media/',
    )
    django.setup()

from hypothesis import given, strategies as st, assume
from django.template import Template, Context, TemplateSyntaxError

@given(st.text(min_size=1, max_size=50))
def test_prefix_node_malformed_as_clause(text):
    assume('"' not in text and "'" not in text)
    assume(' ' not in text and text != 'as')

    # Testing get_static_prefix with malformed 'as' clause
    template_str = "{% load static %}{% get_static_prefix as %}"

    try:
        template = Template(template_str)
    except TemplateSyntaxError:
        # This is the expected behavior
        pass
    except IndexError:
        # This is the bug - should raise TemplateSyntaxError instead
        assert False, "Should raise TemplateSyntaxError, not IndexError"

# Run the property-based test
if __name__ == "__main__":
    print("Running property-based test for PrefixNode bug...")
    print("Testing template: {% get_static_prefix as %}")
    print()

    try:
        test_prefix_node_malformed_as_clause()
        print("✗ Test FAILED: Bug confirmed - IndexError raised instead of TemplateSyntaxError")
    except AssertionError as e:
        print(f"✗ Test FAILED: {e}")
        print("\nThis confirms the bug - Django raises IndexError when it should raise TemplateSyntaxError")
    except Exception as e:
        print(f"✗ Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
    else:
        print("✓ Test passed - no bug found (unexpected!)")
```

<details>

<summary>
**Failing input**: `{% get_static_prefix as %}`
</summary>
```
Running property-based test for PrefixNode bug...
Testing template: {% get_static_prefix as %}

✗ Test FAILED: Should raise TemplateSyntaxError, not IndexError

This confirms the bug - Django raises IndexError when it should raise TemplateSyntaxError
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal test case demonstrating the IndexError bug in django.templatetags.static.PrefixNode
"""

import os
import sys
import django
from django.conf import settings

# Configure minimal Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': False,
            'OPTIONS': {
                'context_processors': [],
            },
        }],
        INSTALLED_APPS=[
            'django.contrib.staticfiles',
        ],
        STATIC_URL='/static/',
        MEDIA_URL='/media/',
    )
    django.setup()

from django.template import Template, Context, TemplateSyntaxError

# Test case 1: get_static_prefix with 'as' but no variable name
print("=" * 60)
print("Test 1: {% get_static_prefix as %}")
print("=" * 60)
template_str = "{% load static %}{% get_static_prefix as %}"

try:
    template = Template(template_str)
    print("✓ Template compiled successfully (unexpected!)")
except TemplateSyntaxError as e:
    print(f"✓ Correctly raised TemplateSyntaxError: {e}")
except IndexError as e:
    print(f"✗ BUG: IndexError raised instead of TemplateSyntaxError")
    print(f"  Error message: {e}")
    import traceback
    print(f"  Traceback:")
    traceback.print_exc()

print()

# Test case 2: get_media_prefix with 'as' but no variable name
print("=" * 60)
print("Test 2: {% get_media_prefix as %}")
print("=" * 60)
template_str = "{% load static %}{% get_media_prefix as %}"

try:
    template = Template(template_str)
    print("✓ Template compiled successfully (unexpected!)")
except TemplateSyntaxError as e:
    print(f"✓ Correctly raised TemplateSyntaxError: {e}")
except IndexError as e:
    print(f"✗ BUG: IndexError raised instead of TemplateSyntaxError")
    print(f"  Error message: {e}")
    import traceback
    print(f"  Traceback:")
    traceback.print_exc()

print()

# Test case 3: Valid usage without 'as' clause (should work)
print("=" * 60)
print("Test 3: {% get_static_prefix %} (valid usage)")
print("=" * 60)
template_str = "{% load static %}{% get_static_prefix %}"

try:
    template = Template(template_str)
    result = template.render(Context())
    print(f"✓ Template compiled and rendered successfully")
    print(f"  Result: '{result}'")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print()

# Test case 4: Valid usage with 'as' clause (should work)
print("=" * 60)
print("Test 4: {% get_static_prefix as my_prefix %} (valid usage)")
print("=" * 60)
template_str = "{% load static %}{% get_static_prefix as my_prefix %}{{ my_prefix }}"

try:
    template = Template(template_str)
    result = template.render(Context())
    print(f"✓ Template compiled and rendered successfully")
    print(f"  Result: '{result}'")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
```

<details>

<summary>
IndexError raised when parsing malformed template tags
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/repo.py", line 40, in <module>
    template = Template(template_str)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 154, in __init__
    self.nodelist = self.compile_nodelist()
                    ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 196, in compile_nodelist
    nodelist = parser.parse()
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 518, in parse
    raise self.error(token, e)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 516, in parse
    compiled_result = compile_func(self, token)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/templatetags/static.py", line 74, in get_static_prefix
    return PrefixNode.handle_token(parser, token, "STATIC_URL")
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/templatetags/static.py", line 36, in handle_token
    varname = tokens[2]
              ~~~~~~^^^
IndexError: list index out of range
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/repo.py", line 60, in <module>
    template = Template(template_str)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 154, in __init__
    self.nodelist = self.compile_nodelist()
                    ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 196, in compile_nodelist
    nodelist = parser.parse()
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 518, in parse
    raise self.error(token, e)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 516, in parse
    compiled_result = compile_func(self, token)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/templatetags/static.py", line 92, in get_media_prefix
    return PrefixNode.handle_token(parser, token, "MEDIA_URL")
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/templatetags/static.py", line 36, in handle_token
    varname = tokens[2]
              ~~~~~~^^^
IndexError: list index out of range
============================================================
Test 1: {% get_static_prefix as %}
============================================================
✗ BUG: IndexError raised instead of TemplateSyntaxError
  Error message: list index out of range
  Traceback:

============================================================
Test 2: {% get_media_prefix as %}
============================================================
✗ BUG: IndexError raised instead of TemplateSyntaxError
  Error message: list index out of range
  Traceback:

============================================================
Test 3: {% get_static_prefix %} (valid usage)
============================================================
✓ Template compiled and rendered successfully
  Result: '/static/'

============================================================
Test 4: {% get_static_prefix as my_prefix %} (valid usage)
============================================================
✓ Template compiled and rendered successfully
  Result: '/static/'
```
</details>

## Why This Is A Bug

Django's template system should always raise `TemplateSyntaxError` for malformed template syntax, not low-level Python exceptions like `IndexError`. This violates Django's documented error handling conventions in several ways:

1. **Django's Template Tag Documentation**: According to Django's custom template tags documentation, template tag compilation functions should "raise template.TemplateSyntaxError" for syntax errors with helpful, descriptive messages.

2. **Inconsistent Error Handling**: The code already properly validates that the second token must be "as" (lines 31-34), but fails to validate that a third token (the variable name) exists when "as" is present.

3. **Poor Developer Experience**: The error message "list index out of range" provides no context about what's wrong with the template syntax. A proper `TemplateSyntaxError` would explain that a variable name is required after 'as'.

4. **Logic Error in Bounds Checking**: At line 35, the code checks `if len(tokens) > 1` but then immediately accesses `tokens[2]` at line 36. When there are exactly 2 tokens (e.g., `['get_static_prefix', 'as']`), this check passes but `tokens[2]` doesn't exist, causing the IndexError.

## Relevant Context

The bug affects both `get_static_prefix` and `get_media_prefix` template tags since they both use the same `PrefixNode.handle_token` method defined in `/django/templatetags/static.py`.

The docstrings for these tags (lines 65-72 and 83-90) show the valid usage patterns:
- `{% get_static_prefix %}` - Returns the prefix directly
- `{% get_static_prefix as varname %}` - Stores the prefix in a variable

The bug occurs when a developer accidentally writes `{% get_static_prefix as %}` without the variable name, which is a common typo.

Django source code location: https://github.com/django/django/blob/main/django/templatetags/static.py

## Proposed Fix

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
+                "'%s as' requires a variable name" % tokens[0]
+            )
         return cls(varname, name)
```