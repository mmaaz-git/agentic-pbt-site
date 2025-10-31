# Bug Report: django.core.checks.security.check_referrer_policy Empty String Validation

**Target**: `django.core.checks.security.base.check_referrer_policy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_referrer_policy` function incorrectly rejects valid referrer policy strings that contain trailing commas or consecutive commas, due to improper handling of empty strings created during comma-separated value parsing.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings as hyp_settings
from django.core.checks.security.base import (
    check_referrer_policy,
    REFERRER_POLICY_VALUES,
)

valid_policies = list(REFERRER_POLICY_VALUES)

@hyp_settings(max_examples=300)
@given(
    st.lists(st.sampled_from(valid_policies), min_size=1, max_size=3),
    st.integers(min_value=1, max_value=5),
)
def test_trailing_commas_should_not_error(policies, num_trailing_commas):
    settings.MIDDLEWARE = ['django.middleware.security.SecurityMiddleware']

    policy_string = ', '.join(policies) + ',' * num_trailing_commas
    settings.SECURE_REFERRER_POLICY = policy_string

    result = check_referrer_policy(None)

    assert result == [], \
        f"Valid policies with trailing commas {policy_string!r} should not error, got: {result}"
```

**Failing input**: `policies=['origin'], num_trailing_commas=1` (resulting in string `"origin,"`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.conf import settings
from django.core.checks.security.base import check_referrer_policy, E023

settings.MIDDLEWARE = ['django.middleware.security.SecurityMiddleware']
settings.SECURE_REFERRER_POLICY = "origin,"

result = check_referrer_policy(None)
print(result)
```

Expected: `[]` (no errors)
Actual: `[<Error: ... id='security.E023'>]` (validation error)

The same bug occurs with double commas: `"origin,,same-origin"` also incorrectly fails validation.

## Why This Is A Bug

When a user specifies a valid referrer policy like `"origin,"` or `"origin,,same-origin"`, the current implementation:

1. Splits by comma: `"origin,".split(",")` → `['origin', '']`
2. Strips each value: `{v.strip() for v in ['origin', '']}` → `{'origin', ''}`
3. Checks subset membership: `{'origin', ''} <= REFERRER_POLICY_VALUES` → `False` (because `''` is not valid)
4. Returns error E023

However, trailing/consecutive commas are common formatting patterns that should be tolerated. The empty strings created during parsing are not meaningful policy values and should be filtered out, not treated as invalid policies.

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

The fix filters out empty strings by adding `if v.strip()` to the set comprehension, ensuring only non-empty values are validated.