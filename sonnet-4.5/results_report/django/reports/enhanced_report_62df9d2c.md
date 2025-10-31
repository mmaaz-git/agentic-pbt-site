# Bug Report: Django TemplateCommand Camel Case Conversion with Digits

**Target**: `django.core.management.templates.TemplateCommand.handle()` (line 138)
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The camel case conversion for Django app/project names containing digits produces unexpected capitalization patterns, treating digits as word boundaries and capitalizing the letter following each digit.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for Django TemplateCommand camel case conversion bug.
Tests that letters after digits should not be unexpectedly capitalized.
"""

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
    """Test that digits don't cause unexpected capitalization of following letters."""
    result = to_camel_case(name)

    for i in range(len(name) - 1):
        if name[i].isdigit() and name[i+1].isalpha():
            title_name = name.title()
            is_uppercase_after_digit = title_name[i+1].isupper()

            if is_uppercase_after_digit and "_" not in name[:i+2]:
                assert False, f"Bug: '{name}' -> '{result}' - letter after digit unexpectedly capitalized"


if __name__ == "__main__":
    # Run the test
    test_camel_case_unexpected_capitals_after_digits()
```

<details>

<summary>
**Failing input**: `my2app`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 36, in <module>
  |     test_camel_case_unexpected_capitals_after_digits()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 17, in test_camel_case_unexpected_capitals_after_digits
  |     @example("my2app")
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 31, in test_camel_case_unexpected_capitals_after_digits
    |     assert False, f"Bug: '{name}' -> '{result}' - letter after digit unexpectedly capitalized"
    |            ^^^^^
    | AssertionError: Bug: 'my2app' -> 'My2App' - letter after digit unexpectedly capitalized
    | Falsifying explicit example: test_camel_case_unexpected_capitals_after_digits(
    |     name='my2app',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 31, in test_camel_case_unexpected_capitals_after_digits
    |     assert False, f"Bug: '{name}' -> '{result}' - letter after digit unexpectedly capitalized"
    |            ^^^^^
    | AssertionError: Bug: 'app2api' -> 'App2Api' - letter after digit unexpectedly capitalized
    | Falsifying explicit example: test_camel_case_unexpected_capitals_after_digits(
    |     name='app2api',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 31, in test_camel_case_unexpected_capitals_after_digits
    |     assert False, f"Bug: '{name}' -> '{result}' - letter after digit unexpectedly capitalized"
    |            ^^^^^
    | AssertionError: Bug: 'test1module' -> 'Test1Module' - letter after digit unexpectedly capitalized
    | Falsifying explicit example: test_camel_case_unexpected_capitals_after_digits(
    |     name='test1module',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of Django TemplateCommand camel case conversion bug
with digits in app/project names.
"""

def to_camel_case(name):
    """From Django's management/templates.py line 138"""
    return "".join(x for x in name.title() if x != "_")


# Main test case
print("=== Django Camel Case Conversion Bug ===")
print()
print("Input: 'my2app'")
print(f"Step 1 - .title(): {'my2app'.title()}")
print(f"Step 2 - remove underscores: {to_camel_case('my2app')}")
print(f"Expected: 'My2app' (digit doesn't create word boundary)")
print(f"Actual: 'My2App' (letter after digit is capitalized)")
print()
print("ERROR: The 'a' after '2' is incorrectly capitalized")
print()

# Additional examples
print("=== Additional Examples ===")
examples = [
    ("test1module", "Test1module", "Test1Module"),
    ("app2api", "App2api", "App2Api"),
    ("my3rd_app", "My3rdApp", "My3RdApp"),
    ("version2_0", "Version20", "Version20"),  # This one works as expected
    ("my_2_app", "My2App", "My2App"),  # This one works as expected
]

print("Input         -> Expected     -> Actual       | Correct?")
print("-" * 55)
for input_name, expected, actual in examples:
    result = to_camel_case(input_name)
    is_correct = result == expected
    print(f"{input_name:13} -> {expected:12} -> {result:12} | {'✓' if is_correct else '✗'}")
    if result != actual:
        print(f"  WARNING: Got '{result}' but example shows '{actual}'")

print()
print("=== Analysis ===")
print("Python's .title() method treats digits as word boundaries.")
print("This causes unexpected capitalization for valid Python identifiers.")
print("App names with digits (e.g., 'my2app') are valid but produce")
print("counter-intuitive camel case class names (e.g., 'My2AppConfig').")
```

<details>

<summary>
Django Camel Case Conversion Bug - Output
</summary>
```
=== Django Camel Case Conversion Bug ===

Input: 'my2app'
Step 1 - .title(): My2App
Step 2 - remove underscores: My2App
Expected: 'My2app' (digit doesn't create word boundary)
Actual: 'My2App' (letter after digit is capitalized)

ERROR: The 'a' after '2' is incorrectly capitalized

=== Additional Examples ===
Input         -> Expected     -> Actual       | Correct?
-------------------------------------------------------
test1module   -> Test1module  -> Test1Module  | ✗
app2api       -> App2api      -> App2Api      | ✗
my3rd_app     -> My3rdApp     -> My3RdApp     | ✗
version2_0    -> Version20    -> Version20    | ✓
my_2_app      -> My2App       -> My2App       | ✓

=== Analysis ===
Python's .title() method treats digits as word boundaries.
This causes unexpected capitalization for valid Python identifiers.
App names with digits (e.g., 'my2app') are valid but produce
counter-intuitive camel case class names (e.g., 'My2AppConfig').
```
</details>

## Why This Is A Bug

This behavior violates expected Python naming conventions for identifiers containing digits. When developers create Django apps with names like "my2app" (a valid Python identifier), they reasonably expect the camel case conversion to produce "My2app" for the generated `AppConfig` class name, not "My2App".

The issue stems from Python's `str.title()` method treating any non-letter character, including digits, as word boundaries. This creates unexpected capitalization patterns that:

1. Don't match typical Python naming conventions where digits within identifiers don't create word boundaries
2. Produce class names that look incorrect (e.g., `My2AppConfig` instead of `My2appConfig`)
3. Are inconsistent with how most developers manually convert snake_case to CamelCase

While the generated code remains syntactically valid, the naming convention is counterintuitive and may confuse developers or violate project style guides.

## Relevant Context

The bug occurs in Django's template command system used by `startapp` and `startproject` management commands. The affected variable `camel_case_app_name` is used in Django's app templates to generate the default `AppConfig` class name.

Location in Django source: `/django/core/management/templates.py:138`

The Django documentation doesn't specify how camel case conversion should handle digits, leaving this behavior undefined. The current implementation has likely persisted because:
- App names with digits are relatively uncommon
- The generated code still functions correctly
- Developers can manually edit the generated class names

## Proposed Fix

```diff
--- a/django/core/management/templates.py
+++ b/django/core/management/templates.py
@@ -135,7 +135,8 @@ class TemplateCommand(BaseCommand):
         base_subdir = "%s_template" % app_or_project
         base_directory = "%s_directory" % app_or_project
         camel_case_name = "camel_case_%s_name" % app_or_project
-        camel_case_value = "".join(x for x in name.title() if x != "_")
+        # Convert snake_case to CamelCase, splitting only on underscores
+        camel_case_value = "".join(word.capitalize() for word in name.split("_"))

         context = Context(
             {
```