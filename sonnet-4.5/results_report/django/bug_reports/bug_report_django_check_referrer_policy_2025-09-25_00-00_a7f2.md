# Bug Report: django.core.checks.security.base.check_referrer_policy Empty String Validation

**Target**: `django.core.checks.security.base.check_referrer_policy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_referrer_policy` function incorrectly rejects valid referrer policy strings that contain trailing commas, double commas, or commas with only whitespace. This causes false positive security check failures.

## Property-Based Test

```python
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
```

**Failing input**: `"no-referrer,"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/django')

from django.core.checks.security.base import REFERRER_POLICY_VALUES

test_string = "no-referrer,"
values = {v.strip() for v in test_string.split(",")}

print(f"Input: {test_string!r}")
print(f"Parsed: {values}")
print(f"Contains empty string: {'' in values}")
print(f"Is valid subset: {values <= REFERRER_POLICY_VALUES}")
```

**Output:**
```
Input: 'no-referrer,'
Parsed: {'no-referrer', ''}
Contains empty string: True
Is valid subset: False
```

## Why This Is A Bug

The function is documented (line 264) as supporting "a comma-separated string" for referrer policy values. However, the current implementation does not handle common edge cases in comma-separated lists:

1. Trailing commas: `"no-referrer,"`
2. Double commas: `"no-referrer,,same-origin"`
3. Commas with only whitespace: `"no-referrer, ,same-origin"`

These patterns are common in configuration files and should be accepted. The parsing logic creates empty strings in the `values` set, which are not in `REFERRER_POLICY_VALUES`, causing the check to incorrectly fail with error E023.

This affects users who:
- Use configuration templates that may add trailing commas
- Format their comma-separated lists with extra whitespace
- Have automated configuration generation tools

## Fix

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