#!/usr/bin/env python3
"""
Minimal reproduction of Django CSRF_TRUSTED_ORIGINS validation bug.
This demonstrates that malformed URLs pass the check but produce
empty components when parsed, breaking the CSRF middleware.
"""

import sys
import os
# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings first
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
import django
from django.conf import settings
settings.configure(
    DEBUG=True,
    SECRET_KEY='test-secret-key',
    CSRF_TRUSTED_ORIGINS=[]
)

from unittest.mock import patch
from urllib.parse import urlsplit

# Import the check function
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

# Test cases of malformed URLs that should fail validation
malformed_origins = [
    "://",
    "://example.com",
    "http://",
    "https://"
]

print("Django CSRF_TRUSTED_ORIGINS Validation Bug Demonstration")
print("=" * 60)

for origin in malformed_origins:
    print(f"\nTesting origin: '{origin}'")
    print("-" * 40)

    # Test the check function
    with patch.object(settings, 'CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(None)

        passes_check = len(errors) == 0
        print(f"  Passes validation check: {passes_check}")

        if errors:
            print(f"  Error message: {errors[0].msg}")

    # Show what urlsplit actually produces
    parsed = urlsplit(origin)
    print(f"  urlsplit() results:")
    print(f"    scheme: '{parsed.scheme}' (empty={not parsed.scheme})")
    print(f"    netloc: '{parsed.netloc}' (empty={not parsed.netloc})")

    # Show the impact on middleware
    if passes_check:
        print(f"  PROBLEM: This malformed URL passes validation but:")
        if not parsed.scheme:
            print(f"    - Has no scheme (middleware needs this)")
        if not parsed.netloc:
            print(f"    - Has no netloc (middleware needs this)")
        print(f"    - Would cause silent CSRF protection failure")

print("\n" + "=" * 60)
print("CONCLUSION: The check function incorrectly accepts malformed URLs")
print("that cannot be properly parsed by the CSRF middleware.")