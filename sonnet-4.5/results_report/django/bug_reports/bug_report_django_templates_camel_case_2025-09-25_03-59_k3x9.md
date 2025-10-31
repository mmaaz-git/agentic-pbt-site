# Bug Report: Django TemplateCommand Camel Case Conversion with Digits

**Target**: `django.core.management.templates.TemplateCommand` (line 138)
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The camel case conversion for project/app names containing digits produces unexpected capitalization. When creating an app or project with a name containing digits (e.g., "my2app"), the `.title()` method treats digits as word boundaries, causing the letter after each digit to be capitalized incorrectly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
import string


def to_camel_case(name):
    """Convert name to camel case as done in Django templates.py line 138."""
    return "".join(x for x in name.title() if x != "_")


@given(st.text(alphabet=string.ascii_lowercase + string.digits, min_size=3, max_size=20))
@example("my2app")
@example("app2api")
@example("test1module")
@settings(max_examples=500)
def test_camel_case_unexpected_capitals_after_digits(name):
    result = to_camel_case(name)

    for i in range(len(name) - 1):
        if name[i].isdigit() and name[i+1].isalpha():
            title_name = name.title()
            is_uppercase_after_digit = title_name[i+1].isupper()

            if is_uppercase_after_digit and "_" not in name[:i+2]:
                assert False, f"Bug: '{name}' -> '{result}' - letter after digit unexpectedly capitalized"
```

**Failing input**: `my2app`

## Reproducing the Bug

```python
def to_camel_case(name):
    """From Django's management/templates.py line 138"""
    return "".join(x for x in name.title() if x != "_")


print("Input: 'my2app'")
print(f"Step 1 - .title(): {'my2app'.title()}")
print(f"Step 2 - remove underscores: {to_camel_case('my2app')}")
print(f"Expected: 'My2app'")
print(f"Actual: 'My2App'")
print()
print("The 'a' after '2' is incorrectly capitalized")

print("\nMore examples:")
examples = ["test1module", "app2api", "my_2_app"]
for ex in examples:
    result = to_camel_case(ex)
    print(f"  '{ex}' -> '{result}'")
```

**Output:**
```
Input: 'my2app'
Step 1 - .title(): My2App
Step 2 - remove underscores: My2App
Expected: My2app
Actual: My2App

The 'a' after '2' is incorrectly capitalized

More examples:
  'test1module' -> 'Test1Module'
  'app2api' -> 'App2Api'
  'my_2_app' -> 'My2App'
```

## Why This Is A Bug

Python's `.title()` method treats any non-letter character (including digits) as a word boundary, causing the next letter to be capitalized. This creates unexpected camel case formatting for valid Python identifiers that contain digits.

For app/project names like "my2app" (which is a valid Python identifier), users would reasonably expect the camel case version to be "My2app", not "My2App". The current implementation makes digits behave like underscores (word separators), which is inconsistent with how digits function in Python identifiers.

## Fix

Replace the `.title()` approach with a proper snake_case to CamelCase conversion:

```diff
--- a/django/core/management/templates.py
+++ b/django/core/management/templates.py
@@ -135,7 +135,8 @@ class TemplateCommand(BaseCommand):
         base_subdir = "%s_template" % app_or_project
         base_directory = "%s_directory" % app_or_project
         camel_case_name = "camel_case_%s_name" % app_or_project
-        camel_case_value = "".join(x for x in name.title() if x != "_")
+        # Convert snake_case to CamelCase, preserving case after digits
+        camel_case_value = "".join(word.capitalize() for word in name.split("_"))

         context = Context(
             {
```

This fix properly handles snake_case by splitting on underscores and capitalizing each word, without treating digits as word boundaries.