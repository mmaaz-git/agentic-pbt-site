# Bug Report: django.core.checks.security check_referrer_policy Empty String Validation

**Target**: `django.core.checks.security.base.check_referrer_policy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_referrer_policy` function incorrectly rejects valid SECURE_REFERRER_POLICY configurations when they contain trailing or leading commas, or any whitespace that results in empty strings after splitting.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.checks.security.base import check_referrer_policy, REFERRER_POLICY_VALUES
from django.test import override_settings


@given(st.lists(st.sampled_from(list(REFERRER_POLICY_VALUES)), min_size=1, max_size=3))
def test_trailing_comma_equivalence(values):
    comma_string_normal = ",".join(values)
    comma_string_trailing = comma_string_normal + ","

    with override_settings(
        MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
        SECURE_REFERRER_POLICY=comma_string_normal
    ):
        result_normal = check_referrer_policy(None)

    with override_settings(
        MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
        SECURE_REFERRER_POLICY=comma_string_trailing
    ):
        result_trailing = check_referrer_policy(None)

    assert result_normal == result_trailing
```

**Failing input**: `["no-referrer"]` which produces `"no-referrer,"` with trailing comma

## Reproducing the Bug

```python
import django
django.setup()

from django.core.checks.security.base import check_referrer_policy
from django.test import override_settings

with override_settings(
    MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
    SECURE_REFERRER_POLICY="no-referrer"
):
    result = check_referrer_policy(None)
    assert len(result) == 0

with override_settings(
    MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
    SECURE_REFERRER_POLICY="no-referrer,"
):
    result = check_referrer_policy(None)
    assert len(result) == 1
    assert result[0].id == "security.E023"
```

## Why This Is A Bug

According to the code comment on line 264 of base.py, the function is designed to "Support a comma-separated string or iterable of values to allow fallback." However, the implementation does not properly handle common edge cases like trailing commas, which are easily introduced during configuration or programmatic string construction.

When `SECURE_REFERRER_POLICY = "no-referrer,"`, the code splits by comma to get `['no-referrer', '']`, and after stripping whitespace, creates the set `{'no-referrer', ''}`. The empty string is not in `REFERRER_POLICY_VALUES`, causing validation to fail with error E023.

This makes the API fragile and rejects configurations that should be valid. Users expect trailing commas to be handled gracefully, especially since:
1. Python allows trailing commas in lists/tuples
2. Many configuration formats ignore trailing commas
3. The semantic meaning is clear (the empty value should be ignored)

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

This change filters out empty strings after stripping, making the parsing robust against trailing/leading commas and extra whitespace.