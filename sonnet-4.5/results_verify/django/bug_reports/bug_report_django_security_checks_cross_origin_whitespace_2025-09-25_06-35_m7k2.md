# Bug Report: django.core.checks.security Cross-Origin Opener Policy Whitespace Bug

**Target**: `django.core.checks.security.base.check_cross_origin_opener_policy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_cross_origin_opener_policy` function incorrectly rejects valid cross-origin opener policy values when they contain leading or trailing whitespace, producing false positive errors.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from django.core.checks.security import base
from django.conf import settings as django_settings

@given(
    st.sampled_from(list(base.CROSS_ORIGIN_OPENER_POLICY_VALUES)),
    st.text(alphabet=' \t\n', min_size=0, max_size=5)
)
@settings(max_examples=50)
def test_cross_origin_opener_policy_no_whitespace_handling(valid_value, whitespace):
    assume(len(whitespace) > 0)

    django_settings.MIDDLEWARE = ['django.middleware.security.SecurityMiddleware']

    django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = valid_value
    result_clean = base.check_cross_origin_opener_policy(None)

    django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = valid_value + whitespace
    result_with_whitespace = base.check_cross_origin_opener_policy(None)

    assert result_clean == result_with_whitespace
```

**Failing input**: `valid_value='unsafe-none'`, `whitespace=' '` (or any whitespace)

## Reproducing the Bug

```python
import os
import django
from django.conf import settings as django_settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

if not django_settings.configured:
    django_settings.configure(
        DEBUG=False,
        SECRET_KEY='test-key-' + 'x' * 50,
        MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
        INSTALLED_APPS=[],
        ALLOWED_HOSTS=['*'],
    )
    django.setup()

from django.core.checks.security.base import check_cross_origin_opener_policy

django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = "unsafe-none"
result_valid = check_cross_origin_opener_policy(None)
print(f"Without whitespace: {result_valid}")

django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = "unsafe-none "
result_with_whitespace = check_cross_origin_opener_policy(None)
print(f"With trailing whitespace: {result_with_whitespace}")
```

Output:
```
Without whitespace: []
With trailing whitespace: [<Error: level=40, msg='You have set the SECURE_CROSS_ORIGIN_OPENER_POLICY setting to an invalid value.', id='security.E024'>]
```

## Why This Is A Bug

1. **Configuration files often have trailing whitespace**: Whether from text editors, copy-paste operations, or multi-line string concatenation, trailing whitespace is common and should be handled gracefully.

2. **Inconsistent with `check_referrer_policy`**: The related `check_referrer_policy` function attempts to handle whitespace by stripping values (though it has its own bug with empty strings). This function should have similar whitespace tolerance.

3. **The setting value is semantically identical**: `"unsafe-none"` and `"unsafe-none "` represent the same policy value. The whitespace is not meaningful to the HTTP header that will ultimately use this value.

4. **Misleading error message**: Users get an error saying the value is invalid, when in fact `"unsafe-none"` is perfectly valid - it just has extraneous whitespace.

## Fix

The fix is to strip whitespace before validation. In `django/core/checks/security/base.py` at lines 275-283:

```diff
 @register(Tags.security, deploy=True)
 def check_cross_origin_opener_policy(app_configs, **kwargs):
     if (
         _security_middleware()
         and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY is not None
-        and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY
+        and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY.strip()
         not in CROSS_ORIGIN_OPENER_POLICY_VALUES
     ):
         return [E024]
     return []
```

Note: This assumes `SECURE_CROSS_ORIGIN_OPENER_POLICY` is always a string when not None. If it could be other types, additional type checking may be needed.