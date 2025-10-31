#!/usr/bin/env python3
"""Property-based test for CSRF_TRUSTED_ORIGINS check"""

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

from hypothesis import given, settings as hyp_settings, strategies as st
from unittest.mock import patch
from urllib.parse import urlsplit
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

@given(st.builds(lambda prefix, suffix: prefix + "://" + suffix, st.text(), st.text()))
@hyp_settings(max_examples=1000)  # Reduced for demo
def test_origins_that_pass_check_are_usable_by_middleware(origin):
    with patch('django.core.checks.compatibility.django_4_0.settings') as mock_settings:
        mock_settings.CSRF_TRUSTED_ORIGINS = [origin]
        errors = check_csrf_trusted_origins(None)

        if len(errors) == 0:
            # The origin passed the check, so it should be usable by middleware
            parsed = urlsplit(origin)
            netloc = parsed.netloc.lstrip("*")

            # Assert that if check passes, the URL should have valid components
            assert parsed.scheme, f"Origin '{origin}' has no scheme but passed check"
            assert netloc, f"Origin '{origin}' has no netloc but passed check"

def test_specific_origin(origin):
    with patch('django.core.checks.compatibility.django_4_0.settings') as mock_settings:
        mock_settings.CSRF_TRUSTED_ORIGINS = [origin]
        errors = check_csrf_trusted_origins(None)

        if len(errors) == 0:
            # The origin passed the check, so it should be usable by middleware
            parsed = urlsplit(origin)
            netloc = parsed.netloc.lstrip("*")

            # Assert that if check passes, the URL should have valid components
            assert parsed.scheme, f"Origin '{origin}' has no scheme but passed check"
            assert netloc, f"Origin '{origin}' has no netloc but passed check"

if __name__ == "__main__":
    # Try some specific cases first
    print("Testing specific malformed origins...")
    test_cases = ["://", "://example.com", "http://", "https://"]

    for origin in test_cases:
        try:
            test_specific_origin(origin)
            print(f"✓ Origin '{origin}' handled correctly")
        except AssertionError as e:
            print(f"✗ {e}")

    print("\nRunning property-based tests with Hypothesis...")
    import pytest
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])