# Bug Report: django.core.checks.security Cross-Origin Opener Policy Whitespace Validation Failure

**Target**: `django.core.checks.security.base.check_cross_origin_opener_policy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_cross_origin_opener_policy` function incorrectly rejects valid Cross-Origin Opener Policy values when they contain leading or trailing whitespace, producing misleading validation errors for semantically correct configuration values.

## Property-Based Test

```python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEBUG=False,
        SECRET_KEY='test-key-' + 'x' * 50,
        MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
        INSTALLED_APPS=[],
        ALLOWED_HOSTS=['*'],
    )
    django.setup()

from hypothesis import given, strategies as st, settings, assume
from django.core.checks.security import base

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

    assert result_clean == result_with_whitespace, f"Results differ for '{valid_value}' with whitespace '{repr(whitespace)}'"

if __name__ == "__main__":
    test_cross_origin_opener_policy_no_whitespace_handling()
```

<details>

<summary>
**Failing input**: `valid_value='same-origin-allow-popups'`, `whitespace=' '`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 40, in <module>
    test_cross_origin_opener_policy_no_whitespace_handling()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 22, in test_cross_origin_opener_policy_no_whitespace_handling
    st.sampled_from(list(base.CROSS_ORIGIN_OPENER_POLICY_VALUES)),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 37, in test_cross_origin_opener_policy_no_whitespace_handling
    assert result_clean == result_with_whitespace, f"Results differ for '{valid_value}' with whitespace '{repr(whitespace)}'"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Results differ for 'same-origin-allow-popups' with whitespace '' ''
Falsifying example: test_cross_origin_opener_policy_no_whitespace_handling(
    # The test always failed when commented parts were varied together.
    valid_value='same-origin-allow-popups',  # or any other generated value
    whitespace=' ',  # or any other generated value
)
```
</details>

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

# Test without whitespace
django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = "unsafe-none"
result_valid = check_cross_origin_opener_policy(None)
print(f"Without whitespace: {result_valid}")

# Test with trailing whitespace
django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = "unsafe-none "
result_with_whitespace = check_cross_origin_opener_policy(None)
print(f"With trailing whitespace: {result_with_whitespace}")

# Test with leading whitespace
django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = " unsafe-none"
result_with_leading = check_cross_origin_opener_policy(None)
print(f"With leading whitespace: {result_with_leading}")

# Test with both leading and trailing whitespace
django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = " unsafe-none "
result_with_both = check_cross_origin_opener_policy(None)
print(f"With both whitespaces: {result_with_both}")
```

<details>

<summary>
Output showing validation errors for valid values with whitespace
</summary>
```
Without whitespace: []
With trailing whitespace: [<Error: level=40, msg='You have set the SECURE_CROSS_ORIGIN_OPENER_POLICY setting to an invalid value.', hint='Valid values are: same-origin, same-origin-allow-popups, unsafe-none.', obj=None, id='security.E024'>]
With leading whitespace: [<Error: level=40, msg='You have set the SECURE_CROSS_ORIGIN_OPENER_POLICY setting to an invalid value.', hint='Valid values are: same-origin, same-origin-allow-popups, unsafe-none.', obj=None, id='security.E024'>]
With both whitespaces: [<Error: level=40, msg='You have set the SECURE_CROSS_ORIGIN_OPENER_POLICY setting to an invalid value.', hint='Valid values are: same-origin, same-origin-allow-popups, unsafe-none.', obj=None, id='security.E024'>]
```
</details>

## Why This Is A Bug

This violates expected behavior in several important ways:

1. **HTTP Standard Compliance**: RFC 9110 Section 5.5 explicitly states that leading and trailing whitespace in HTTP header field values can be removed without changing the semantic meaning. Django, as a web framework implementing HTTP headers, should follow this standard.

2. **Internal Inconsistency**: The `check_referrer_policy` function in the same module (line 266 of `django/core/checks/security/base.py`) explicitly handles whitespace by calling `.strip()` on configuration values. This creates an inconsistent API where some security header checks handle whitespace gracefully while others don't.

3. **Misleading Error Messages**: The error message states "You have set the SECURE_CROSS_ORIGIN_OPENER_POLICY setting to an invalid value" and lists valid values including the exact value being used (minus whitespace). This is confusing because `"unsafe-none"` IS a valid value - the only issue is extraneous whitespace.

4. **Common Real-World Scenarios**: Configuration values frequently acquire whitespace through copy-paste operations from documentation, multi-line string concatenation in Python settings files, or text editor auto-formatting. Rejecting these creates unnecessary friction in Django deployment.

## Relevant Context

The current implementation at line 275-283 performs a direct membership check:
```python
@register(Tags.security, deploy=True)
def check_cross_origin_opener_policy(app_configs, **kwargs):
    if (
        _security_middleware()
        and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY is not None
        and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY
        not in CROSS_ORIGIN_OPENER_POLICY_VALUES
    ):
        return [E024]
    return []
```

Compare this to the `check_referrer_policy` function immediately above it (line 260-271) which properly handles whitespace:
```python
if isinstance(settings.SECURE_REFERRER_POLICY, str):
    values = {v.strip() for v in settings.SECURE_REFERRER_POLICY.split(",")}
```

The valid COOP values are defined at lines 6-10:
```python
CROSS_ORIGIN_OPENER_POLICY_VALUES = {
    "same-origin",
    "same-origin-allow-popups",
    "unsafe-none",
}
```

Documentation: https://docs.djangoproject.com/en/stable/ref/settings/#secure-cross-origin-opener-policy
HTTP Header Specification: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cross-Origin-Opener-Policy

## Proposed Fix

```diff
@register(Tags.security, deploy=True)
def check_cross_origin_opener_policy(app_configs, **kwargs):
    if (
        _security_middleware()
        and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY is not None
-       and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY
+       and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY.strip()
        not in CROSS_ORIGIN_OPENER_POLICY_VALUES
    ):
        return [E024]
    return []
```