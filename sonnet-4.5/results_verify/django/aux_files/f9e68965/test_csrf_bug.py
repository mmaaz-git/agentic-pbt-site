#!/usr/bin/env python3
"""Test case to reproduce the CSRF_TRUSTED_ORIGINS bug"""

import os
import sys
import django
from django.conf import settings

# Configure Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        CSRF_TRUSTED_ORIGINS=[]
    )
    django.setup()

from unittest.mock import patch
from urllib.parse import urlsplit

# Import the function under test
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

def test_malformed_origins():
    """Test that malformed origins are properly detected"""
    malformed_origins = ["://", "://example.com", "http://"]

    for origin in malformed_origins:
        with patch('django.core.checks.compatibility.django_4_0.settings') as mock_settings:
            mock_settings.CSRF_TRUSTED_ORIGINS = [origin]
            errors = check_csrf_trusted_origins(None)

            print(f"Origin: '{origin}'")
            print(f"  Passes check: {len(errors) == 0}")

            parsed = urlsplit(origin)
            print(f"  scheme: '{parsed.scheme}', netloc: '{parsed.netloc}'")

            # Check what the middleware would see
            if len(errors) == 0:
                # This origin passed the check, but what would middleware do with it?
                netloc = parsed.netloc.lstrip("*")
                print(f"  Middleware would use netloc: '{netloc}'")
                print(f"  Has valid scheme: {bool(parsed.scheme)}")
                print(f"  Has valid netloc: {bool(netloc)}")
                print()

if __name__ == "__main__":
    test_malformed_origins()