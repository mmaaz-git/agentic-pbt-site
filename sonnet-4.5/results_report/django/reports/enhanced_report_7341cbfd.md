# Bug Report: django.core.checks.security Empty Referrer Policy List Bypasses Security Warning

**Target**: `django.core.checks.security.base.check_referrer_policy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_referrer_policy` function incorrectly allows `SECURE_REFERRER_POLICY = []` to pass validation without warnings, while `SECURE_REFERRER_POLICY = None` correctly triggers warning W022, despite both configurations resulting in no Referrer-Policy header being sent.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test that discovers the bug in Django's check_referrer_policy function.
The function treats empty lists differently from None, even though both result in
no Referrer-Policy header being sent.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings

# Configure Django with minimal settings
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
    """
    Test that None and empty list are treated consistently.
    Both should trigger warnings since neither results in a Referrer-Policy header.
    """
    with override_settings(SECURE_REFERRER_POLICY=policy_value):
        result = check_referrer_policy(None)

    if policy_value is None or policy_value == []:
        assert len(result) > 0, \
            f"Empty policy should trigger warning. Policy: {policy_value}, Result: {result}"


if __name__ == "__main__":
    # Run the test
    test_referrer_policy_none_vs_empty_consistency()
```

<details>

<summary>
**Failing input**: `policy_value=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 48, in <module>
    test_referrer_policy_none_vs_empty_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 29, in test_referrer_policy_none_vs_empty_consistency
    "no-referrer", "no-referrer-when-downgrade", "origin",
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 42, in test_referrer_policy_none_vs_empty_consistency
    assert len(result) > 0, \
           ^^^^^^^^^^^^^^^
AssertionError: Empty policy should trigger warning. Policy: [], Result: []
Falsifying example: test_referrer_policy_none_vs_empty_consistency(
    policy_value=[],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Demonstrates that Django's check_referrer_policy function treats
SECURE_REFERRER_POLICY = [] differently from SECURE_REFERRER_POLICY = None,
even though both result in no Referrer-Policy header being sent.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings

# Configure Django with minimal settings
settings.configure(
    DEBUG=True,
    SECRET_KEY='test-key-for-testing-minimum-length-of-fifty-chars!!',
    MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
)
django.setup()

from django.test import override_settings
from django.core.checks.security.base import check_referrer_policy

print("Testing Django's check_referrer_policy function\n")
print("=" * 60)

# Test with None (should trigger W022 warning)
print("\nTest 1: SECURE_REFERRER_POLICY = None")
print("-" * 40)
with override_settings(SECURE_REFERRER_POLICY=None):
    result_none = check_referrer_policy(None)
    print(f"Number of warnings/errors: {len(result_none)}")
    if result_none:
        for warning in result_none:
            print(f"  - {warning.id}: {warning.msg}")

# Test with empty list (should trigger W022 but doesn't - BUG!)
print("\nTest 2: SECURE_REFERRER_POLICY = []")
print("-" * 40)
with override_settings(SECURE_REFERRER_POLICY=[]):
    result_empty = check_referrer_policy(None)
    print(f"Number of warnings/errors: {len(result_empty)}")
    if result_empty:
        for warning in result_empty:
            print(f"  - {warning.id}: {warning.msg}")
    else:
        print("  No warnings - BUG! Empty list should trigger W022 warning")

# Test with valid value for comparison
print("\nTest 3: SECURE_REFERRER_POLICY = ['same-origin']")
print("-" * 40)
with override_settings(SECURE_REFERRER_POLICY=['same-origin']):
    result_valid = check_referrer_policy(None)
    print(f"Number of warnings/errors: {len(result_valid)}")
    if result_valid:
        for warning in result_valid:
            print(f"  - {warning.id}: {warning.msg}")
    else:
        print("  No warnings - Correct behavior for valid setting")

print("\n" + "=" * 60)
print("\nSUMMARY:")
print(f"  None triggers {len(result_none)} warning(s)")
print(f"  Empty list triggers {len(result_empty)} warning(s)")
print(f"  Valid value triggers {len(result_valid)} warning(s)")
print("\nBUG: Empty list should trigger the same W022 warning as None,")
print("     since both result in no Referrer-Policy header being sent.")
```

<details>

<summary>
Inconsistent warning behavior between None and empty list
</summary>
```
Testing Django's check_referrer_policy function

============================================================

Test 1: SECURE_REFERRER_POLICY = None
----------------------------------------
Number of warnings/errors: 1
  - security.W022: You have not set the SECURE_REFERRER_POLICY setting. Without this, your site will not send a Referrer-Policy header. You should consider enabling this header to protect user privacy.

Test 2: SECURE_REFERRER_POLICY = []
----------------------------------------
Number of warnings/errors: 0
  No warnings - BUG! Empty list should trigger W022 warning

Test 3: SECURE_REFERRER_POLICY = ['same-origin']
----------------------------------------
Number of warnings/errors: 0
  No warnings - Correct behavior for valid setting

============================================================

SUMMARY:
  None triggers 1 warning(s)
  Empty list triggers 0 warning(s)
  Valid value triggers 0 warning(s)

BUG: Empty list should trigger the same W022 warning as None,
     since both result in no Referrer-Policy header being sent.
```
</details>

## Why This Is A Bug

This bug violates Django's security checking contract and creates an inconsistency that can mislead developers:

1. **Inconsistent Security Warning Behavior**: The `check_referrer_policy` function is designed to warn developers when no Referrer-Policy will be sent. The warning W022 explicitly states "You have not set the SECURE_REFERRER_POLICY setting." An empty list functionally means the policy is not set, yet no warning is issued.

2. **Identical Runtime Effect, Different Check Results**: Both `None` and `[]` result in the SecurityMiddleware not sending a Referrer-Policy header (verified in django/middleware/security.py:49, which checks `if self.referrer_policy:` - an empty list is falsy). Despite having the same security impact, only one triggers the warning.

3. **Mathematical Loophole**: The bug occurs because the code converts the empty list to an empty set at line 268 of check_referrer_policy, then checks if `values <= REFERRER_POLICY_VALUES` at line 269. Since an empty set is mathematically a subset of any set, this check passes incorrectly.

4. **Contradicts Django's Security-by-Default Philosophy**: Django provides a secure default value of "same-origin" for SECURE_REFERRER_POLICY. Any configuration that results in no header should be warned about, as it represents a degradation from the secure default.

5. **Developer Confusion**: A developer might set `SECURE_REFERRER_POLICY = []` during configuration (perhaps building a list conditionally that ends up empty) and receive no warning, believing their configuration is correct when in fact no referrer policy protection is active.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/checks/security/base.py` at lines 260-271. The function only checks for `None` explicitly (line 262) but doesn't handle empty collections.

Django's documentation states that SECURE_REFERRER_POLICY can accept:
- A single string value
- A comma-separated string of values
- An iterable (list/tuple) of values

The SecurityMiddleware code (`django/middleware/security.py:49-59`) shows that the header is only set when `self.referrer_policy` is truthy, making both `None` and `[]` functionally equivalent in preventing the header from being sent.

Warning W022 is defined at line 120-125 with the message: "You have not set the SECURE_REFERRER_POLICY setting. Without this, your site will not send a Referrer-Policy header. You should consider enabling this header to protect user privacy."

## Proposed Fix

```diff
--- a/django/core/checks/security/base.py
+++ b/django/core/checks/security/base.py
@@ -259,6 +259,10 @@ def check_allowed_hosts(app_configs, **kwargs):
 @register(Tags.security, deploy=True)
 def check_referrer_policy(app_configs, **kwargs):
     if _security_middleware():
         if settings.SECURE_REFERRER_POLICY is None:
             return [W022]
+        # Empty lists/tuples should also trigger the warning since they
+        # result in no Referrer-Policy header being sent, same as None
+        if isinstance(settings.SECURE_REFERRER_POLICY, (list, tuple)) and len(settings.SECURE_REFERRER_POLICY) == 0:
+            return [W022]
         # Support a comma-separated string or iterable of values to allow fallback.
         if isinstance(settings.SECURE_REFERRER_POLICY, str):
```