# Bug Report: django.core.checks.security check_cross_origin_opener_policy Crashes on Unhashable Types

**Target**: `django.core.checks.security.base.check_cross_origin_opener_policy`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `check_cross_origin_opener_policy` function crashes with `TypeError: unhashable type: 'list'` when `SECURE_CROSS_ORIGIN_OPENER_POLICY` is set to a list or any other unhashable type, instead of returning a validation error.

## Property-Based Test

```python
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings
from django.core.checks.security.base import check_cross_origin_opener_policy


@given(st.lists(st.text(), min_size=0, max_size=3))
@settings(max_examples=100)
def test_cross_origin_opener_policy_with_list(value):
    mock_settings = Mock()
    mock_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = value

    with patch('django.core.checks.security.base.settings', mock_settings):
        with patch('django.core.checks.security.base._security_middleware', return_value=True):
            result = check_cross_origin_opener_policy(None)
```

**Failing input**: `[]` (empty list, or any list)

## Reproducing the Bug

```python
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.checks.security.base import check_cross_origin_opener_policy


mock_settings = Mock()
mock_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = []

with patch('django.core.checks.security.base.settings', mock_settings):
    with patch('django.core.checks.security.base._security_middleware', return_value=True):
        result = check_cross_origin_opener_policy(None)
```

## Why This Is A Bug

The function attempts to check if a value is in a set using the `in` operator, which requires the value to be hashable. When a user mistakenly configures `SECURE_CROSS_ORIGIN_OPENER_POLICY` as a list (e.g., copying the pattern from `SECURE_REFERRER_POLICY` which supports iterables), the function crashes instead of returning a validation error. This violates the expected behavior of Django's security checks, which should gracefully handle configuration errors and report them as check errors, not crash.

## Fix

```diff
--- a/django/core/checks/security/base.py
+++ b/django/core/checks/security/base.py
@@ -275,10 +275,14 @@ def check_referrer_policy(app_configs, **kwargs):

 @register(Tags.security, deploy=True)
 def check_cross_origin_opener_policy(app_configs, **kwargs):
-    if (
-        _security_middleware()
-        and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY is not None
-        and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY
-        not in CROSS_ORIGIN_OPENER_POLICY_VALUES
-    ):
-        return [E024]
+    if _security_middleware() and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY is not None:
+        value = settings.SECURE_CROSS_ORIGIN_OPENER_POLICY
+        # Ensure value is a string (hashable) before checking membership
+        if not isinstance(value, str):
+            return [E024]
+        if value not in CROSS_ORIGIN_OPENER_POLICY_VALUES:
+            return [E024]
     return []
```