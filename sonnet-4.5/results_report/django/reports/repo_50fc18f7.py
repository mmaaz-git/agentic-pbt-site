#!/usr/bin/env python3
"""
Minimal reproduction of Django CSRF trusted origins validation bug.
This demonstrates that origins without a proper scheme are incorrectly accepted.
"""

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

from django.conf import settings
if not settings.configured:
    settings.configure(
        SECRET_KEY='test-secret-key',
        CSRF_TRUSTED_ORIGINS=[],
        SILENCED_SYSTEM_CHECKS=[],
    )

import django
django.setup()

from unittest.mock import patch
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

# Test cases that should all produce errors
test_cases = [
    '://example.com',          # No scheme before ://
    'example.com://foo',       # Scheme not at start
    'example://com',           # :// in middle but not a valid URL format
    '://',                     # Just the separator
]

# Test case that should NOT produce an error
valid_cases = [
    'https://example.com',     # Proper scheme at start
    'http://localhost:8000',   # Proper scheme with port
]

print("=" * 60)
print("Django CSRF Trusted Origins Validation Bug Demonstration")
print("=" * 60)
print()

print("Testing INVALID origins that SHOULD produce errors:")
print("-" * 50)
for origin in test_cases:
    with patch('django.conf.settings.CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(app_configs=None)
        if len(errors) == 0:
            print(f"❌ BUG: '{origin}' - NO ERROR (should fail validation)")
        else:
            print(f"✓ OK: '{origin}' - ERROR generated as expected")
print()

print("Testing VALID origins that should NOT produce errors:")
print("-" * 50)
for origin in valid_cases:
    with patch('django.conf.settings.CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(app_configs=None)
        if len(errors) == 0:
            print(f"✓ OK: '{origin}' - No error as expected")
        else:
            print(f"❌ UNEXPECTED: '{origin}' - ERROR generated")
            print(f"   Error: {errors[0].msg}")

print()
print("=" * 60)
print("CONCLUSION: The validation accepts malformed origins with '://'")
print("anywhere in the string, even without a proper scheme at the start.")
print("=" * 60)