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