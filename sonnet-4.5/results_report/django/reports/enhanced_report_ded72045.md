# Bug Report: django.core.management.utils.handle_extensions Returns Invalid '.' Extension

**Target**: `django.core.management.utils.handle_extensions`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `handle_extensions()` function incorrectly returns `'.'` as a valid file extension when processing comma-separated extension lists that contain empty strings (from double commas, trailing commas, or spaces between commas).

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for django.core.management.utils.handle_extensions."""

from hypothesis import given, strategies as st, settings, example
from django.core.management.utils import handle_extensions

@given(st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz,. ", min_size=1, max_size=30)))
@example(['py,,js'])  # The failing example from the initial report
@settings(max_examples=100)
def test_handle_extensions_no_dot_only_extension(extensions):
    """
    Property: handle_extensions should never return '.' as an extension
    A lone dot is not a valid file extension.
    """
    result = handle_extensions(extensions)
    assert '.' not in result, f"Result contains invalid extension '.': {result}"

if __name__ == "__main__":
    print("Running property-based test for handle_extensions...")
    print("Testing that handle_extensions never returns '.' as an extension")
    print()

    try:
        test_handle_extensions_no_dot_only_extension()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis test checks that handle_extensions should never return '.' as a standalone extension.")
        print("A lone dot is not semantically valid as a file extension.")
```

<details>

<summary>
**Failing input**: `['py,,js']`
</summary>
```
Running property-based test for handle_extensions...
Testing that handle_extensions never returns '.' as an extension

Test failed: Result contains invalid extension '.': {'.', '.js', '.py'}

This test checks that handle_extensions should never return '.' as a standalone extension.
A lone dot is not semantically valid as a file extension.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the django.core.management.utils.handle_extensions bug."""

from django.core.management.utils import handle_extensions

print("Testing handle_extensions with various comma-separated inputs that contain empty strings:")
print()

# Test case 1: Double comma creates empty string
print("Test 1: handle_extensions(['py,,js'])")
result1 = handle_extensions(['py,,js'])
print(f"Result: {result1}")
print(f"Contains '.': {'.' in result1}")
print()

# Test case 2: Trailing comma
print("Test 2: handle_extensions(['py,'])")
result2 = handle_extensions(['py,'])
print(f"Result: {result2}")
print(f"Contains '.': {'.' in result2}")
print()

# Test case 3: Space between commas
print("Test 3: handle_extensions(['py, ,js'])")
result3 = handle_extensions(['py, ,js'])
print(f"Result: {result3}")
print(f"Contains '.': {'.' in result3}")
print()

# Test case 4: Leading comma
print("Test 4: handle_extensions([',py'])")
result4 = handle_extensions([',py'])
print(f"Result: {result4}")
print(f"Contains '.': {'.' in result4}")
print()

# Test case 5: Multiple consecutive commas
print("Test 5: handle_extensions(['py,,,js'])")
result5 = handle_extensions(['py,,,js'])
print(f"Result: {result5}")
print(f"Contains '.': {'.' in result5}")
print()

print("Summary:")
print("The function incorrectly returns '.' as a valid extension when empty strings")
print("are present in the comma-separated input. This happens because empty strings")
print("from split() are prefixed with '.' without checking if they are actually empty.")
```

<details>

<summary>
Function returns '.' as an invalid extension for inputs with empty strings
</summary>
```
Testing handle_extensions with various comma-separated inputs that contain empty strings:

Test 1: handle_extensions(['py,,js'])
Result: {'.js', '.', '.py'}
Contains '.': True

Test 2: handle_extensions(['py,'])
Result: {'.', '.py'}
Contains '.': True

Test 3: handle_extensions(['py, ,js'])
Result: {'.js', '.', '.py'}
Contains '.': True

Test 4: handle_extensions([',py'])
Result: {'.', '.py'}
Contains '.': True

Test 5: handle_extensions(['py,,,js'])
Result: {'.js', '.', '.py'}
Contains '.': True

Summary:
The function incorrectly returns '.' as a valid extension when empty strings
are present in the comma-separated input. This happens because empty strings
from split() are prefixed with '.' without checking if they are actually empty.
```
</details>

## Why This Is A Bug

This violates expected behavior because a standalone dot `'.'` is not a semantically valid file extension. The bug occurs when the function processes empty strings that result from:
- Double commas (`'py,,js'`)
- Trailing commas (`'py,'`)
- Leading commas (`',py'`)
- Spaces between commas (`'py, ,js'`)

The function's docstring shows examples with clean, valid extensions like `'.html'`, `'.js'`, `'.py'`, suggesting empty extensions were not considered. This could cause Django management commands like `makemessages` to unexpectedly match files ending with just a dot (e.g., `file.`), though such files are rare in practice.

The root cause is in lines 47-52 of `/django/core/management/utils.py`: when `split(',')` produces empty strings, the code blindly prefixes them with `'.'` without checking if they're empty, resulting in `'.'` being added to the extension list.

## Relevant Context

The `handle_extensions()` function is used by Django management commands (particularly `makemessages`) to parse file extensions from command-line arguments. Users can specify extensions in multiple ways:
- Multiple `-e` flags: `-e js -e txt`
- Comma-separated: `-e js,txt`
- Mixed: `-e js,txt -e html`

The function is designed to normalize these inputs into a set of extensions with dots (e.g., `{'.js', '.txt', '.html'}`). However, user input often contains typos like trailing commas or double commas, which the current implementation doesn't handle correctly.

Relevant Django documentation: https://docs.djangoproject.com/en/stable/ref/django-admin/#makemessages
Source code location: `/django/core/management/utils.py:34-53`

## Proposed Fix

```diff
--- a/django/core/management/utils.py
+++ b/django/core/management/utils.py
@@ -46,7 +46,7 @@ def handle_extensions(extensions):
     """
     ext_list = []
     for ext in extensions:
-        ext_list.extend(ext.replace(" ", "").split(","))
+        ext_list.extend([e for e in ext.replace(" ", "").split(",") if e])
     for i, ext in enumerate(ext_list):
         if not ext.startswith("."):
             ext_list[i] = ".%s" % ext_list[i]
     return set(ext_list)
```