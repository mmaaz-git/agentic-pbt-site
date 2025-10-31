# Bug Report: django.core.checks.security.base.check_referrer_policy Empty String Validation Failure

**Target**: `django.core.checks.security.base.check_referrer_policy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_referrer_policy` function incorrectly rejects valid referrer policy configurations when the comma-separated string contains trailing commas, double commas, or commas with only whitespace, causing false positive security check failures.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis-based property test for Django check_referrer_policy bug"""

import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.core.checks.security.base import REFERRER_POLICY_VALUES

@given(st.sets(st.sampled_from(list(REFERRER_POLICY_VALUES)), min_size=1, max_size=3))
@settings(max_examples=200)
def test_referrer_policy_trailing_comma(policy_set):
    policy_list = list(policy_set)
    policy_string_with_trailing = ", ".join(policy_list) + ","

    values = {v.strip() for v in policy_string_with_trailing.split(",")}

    assert values <= REFERRER_POLICY_VALUES, \
        f"Trailing comma causes empty string in set: {policy_string_with_trailing!r} -> {values}"

if __name__ == "__main__":
    test_referrer_policy_trailing_comma()
```

<details>

<summary>
**Failing input**: `'unsafe-url,'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 22, in <module>
    test_referrer_policy_trailing_comma()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 11, in test_referrer_policy_trailing_comma
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 18, in test_referrer_policy_trailing_comma
    assert values <= REFERRER_POLICY_VALUES, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Trailing comma causes empty string in set: 'unsafe-url,' -> {'', 'unsafe-url'}
Falsifying example: test_referrer_policy_trailing_comma(
    policy_set={'unsafe-url'},  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstrate the Django check_referrer_policy bug with trailing comma"""

import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from django.core.checks.security.base import REFERRER_POLICY_VALUES

# Test case 1: Trailing comma
test_string1 = "no-referrer,"
values1 = {v.strip() for v in test_string1.split(",")}

print("Test 1: Trailing comma")
print(f"Input: {test_string1!r}")
print(f"Parsed: {values1}")
print(f"Contains empty string: {'' in values1}")
print(f"Is valid subset: {values1 <= REFERRER_POLICY_VALUES}")
print()

# Test case 2: Double comma
test_string2 = "no-referrer,,same-origin"
values2 = {v.strip() for v in test_string2.split(",")}

print("Test 2: Double comma")
print(f"Input: {test_string2!r}")
print(f"Parsed: {values2}")
print(f"Contains empty string: {'' in values2}")
print(f"Is valid subset: {values2 <= REFERRER_POLICY_VALUES}")
print()

# Test case 3: Comma with only whitespace
test_string3 = "no-referrer, ,same-origin"
values3 = {v.strip() for v in test_string3.split(",")}

print("Test 3: Comma with only whitespace")
print(f"Input: {test_string3!r}")
print(f"Parsed: {values3}")
print(f"Contains empty string: {'' in values3}")
print(f"Is valid subset: {values3 <= REFERRER_POLICY_VALUES}")
print()

# Show valid values for reference
print("Valid REFERRER_POLICY_VALUES:")
for value in sorted(REFERRER_POLICY_VALUES):
    print(f"  - {value!r}")
```

<details>

<summary>
Trailing comma, double comma, and whitespace-only values all produce empty strings in the parsed set
</summary>
```
Test 1: Trailing comma
Input: 'no-referrer,'
Parsed: {'', 'no-referrer'}
Contains empty string: True
Is valid subset: False

Test 2: Double comma
Input: 'no-referrer,,same-origin'
Parsed: {'', 'same-origin', 'no-referrer'}
Contains empty string: True
Is valid subset: False

Test 3: Comma with only whitespace
Input: 'no-referrer, ,same-origin'
Parsed: {'', 'same-origin', 'no-referrer'}
Contains empty string: True
Is valid subset: False

Valid REFERRER_POLICY_VALUES:
  - 'no-referrer'
  - 'no-referrer-when-downgrade'
  - 'origin'
  - 'origin-when-cross-origin'
  - 'same-origin'
  - 'strict-origin'
  - 'strict-origin-when-cross-origin'
  - 'unsafe-url'
```
</details>

## Why This Is A Bug

The function `check_referrer_policy` is documented on line 264 of `/django/core/checks/security/base.py` as supporting "a comma-separated string or iterable of values to allow fallback." However, the current implementation on line 266 splits the string and strips whitespace from each element without filtering out empty strings:

```python
values = {v.strip() for v in settings.SECURE_REFERRER_POLICY.split(",")}
```

This parsing logic creates empty strings in the `values` set when:
1. There's a trailing comma at the end of the string (e.g., `"no-referrer,"`)
2. There are consecutive commas (e.g., `"no-referrer,,same-origin"`)
3. There are commas with only whitespace between them (e.g., `"no-referrer, ,same-origin"`)

Since the empty string `''` is not in `REFERRER_POLICY_VALUES`, the subset check on line 269 (`values <= REFERRER_POLICY_VALUES`) fails, causing the function to incorrectly return error E023 ("You have set the SECURE_REFERRER_POLICY setting to an invalid value").

This violates expected behavior because:
- Trailing commas are a common pattern in configuration files and lists
- Many configuration generation tools and templates produce trailing commas
- The documentation doesn't specify that trailing commas are invalid
- The error message suggests the *values* are invalid, when actually the values themselves are correct

## Relevant Context

The `SECURE_REFERRER_POLICY` setting in Django controls the `Referrer-Policy` HTTP header, which browsers use to determine what referrer information should be included with requests. The setting accepts either:
- A single string value from `REFERRER_POLICY_VALUES`
- A comma-separated string of values (for fallback support)
- An iterable of values

The valid referrer policy values defined in Django are:
- `'no-referrer'`
- `'no-referrer-when-downgrade'`
- `'origin'`
- `'origin-when-cross-origin'`
- `'same-origin'`
- `'strict-origin'`
- `'strict-origin-when-cross-origin'`
- `'unsafe-url'`

Django documentation: https://docs.djangoproject.com/en/stable/ref/settings/#secure-referrer-policy
Source code: `/django/core/checks/security/base.py` lines 260-271

## Proposed Fix

```diff
--- a/django/core/checks/security/base.py
+++ b/django/core/checks/security/base.py
@@ -263,7 +263,7 @@ def check_referrer_policy(app_configs, **kwargs):
             return [W022]
         # Support a comma-separated string or iterable of values to allow fallback.
         if isinstance(settings.SECURE_REFERRER_POLICY, str):
-            values = {v.strip() for v in settings.SECURE_REFERRER_POLICY.split(",")}
+            values = {v.strip() for v in settings.SECURE_REFERRER_POLICY.split(",") if v.strip()}
         else:
             values = set(settings.SECURE_REFERRER_POLICY)
         if not values <= REFERRER_POLICY_VALUES:
```