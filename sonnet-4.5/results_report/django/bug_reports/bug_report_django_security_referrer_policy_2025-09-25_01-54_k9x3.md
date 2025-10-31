# Bug Report: django.core.checks.security Empty Referrer Policy List Bypasses Warning

**Target**: `django.core.checks.security.base.check_referrer_policy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_referrer_policy` function treats `SECURE_REFERRER_POLICY = []` (empty list) differently from `SECURE_REFERRER_POLICY = None`, allowing an empty referrer policy to pass validation without any warnings. This is inconsistent and potentially misleading, as both represent "no referrer policy configured" and should warn the user about the security implications.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')
import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-key-for-testing-minimum-length-of-fifty-chars!!',
        MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
    )
    django.setup()

from hypothesis import given, strategies as st
from django.test import override_settings
from django.core.checks.security.base import check_referrer_policy


@given(st.one_of(st.none(), st.lists(st.sampled_from([
    "no-referrer", "no-referrer-when-downgrade", "origin",
    "origin-when-cross-origin", "same-origin", "strict-origin",
    "strict-origin-when-cross-origin", "unsafe-url"
]), min_size=0, max_size=5)))
def test_referrer_policy_none_vs_empty_consistency(policy_value):
    with override_settings(SECURE_REFERRER_POLICY=policy_value):
        result = check_referrer_policy(None)

    if policy_value is None or policy_value == []:
        assert len(result) > 0, \
            f"Empty policy should trigger warning. Policy: {policy_value}, Result: {result}"
```

**Failing input**: `policy_value=[]`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')
import django
from django.conf import settings
settings.configure(
    DEBUG=True,
    SECRET_KEY='test' * 20,
    MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
)
django.setup()

from django.test import override_settings
from django.core.checks.security.base import check_referrer_policy

with override_settings(SECURE_REFERRER_POLICY=None):
    result_none = check_referrer_policy(None)
    print(f"None: {len(result_none)} warnings")

with override_settings(SECURE_REFERRER_POLICY=[]):
    result_empty = check_referrer_policy(None)
    print(f"[]:   {len(result_empty)} warnings")
```

Output:
```
None: 1 warnings
[]:   0 warnings
```

## Why This Is A Bug

The current implementation has inconsistent behavior:

1. `SECURE_REFERRER_POLICY = None` → Returns W022 warning (correct behavior)
2. `SECURE_REFERRER_POLICY = []` → Returns no warnings (BUG!)

Both `None` and an empty list represent "no referrer policy configured" from a security perspective. The empty list case passes validation because the code converts it to an empty set `set([])`, and an empty set is mathematically a subset of any set, so the check `values <= REFERRER_POLICY_VALUES` passes.

This is problematic because:
- A developer might accidentally set `SECURE_REFERRER_POLICY = []` thinking they've configured it
- The security check silently passes, giving false confidence
- The site runs without a referrer policy header, exposing user privacy

The check should treat empty lists the same as `None` and warn the user.

## Fix

```diff
--- a/django/core/checks/security/base.py
+++ b/django/core/checks/security/base.py
@@ -281,6 +281,8 @@ def check_referrer_policy(app_configs, **kwargs):
     if _security_middleware():
         if settings.SECURE_REFERRER_POLICY is None:
             return [W022]
+        if isinstance(settings.SECURE_REFERRER_POLICY, (list, tuple)) and len(settings.SECURE_REFERRER_POLICY) == 0:
+            return [W022]
         # Support a comma-separated string or iterable of values to allow fallback.
         if isinstance(settings.SECURE_REFERRER_POLICY, str):
             values = {v.strip() for v in settings.SECURE_REFERRER_POLICY.split(",")}
```