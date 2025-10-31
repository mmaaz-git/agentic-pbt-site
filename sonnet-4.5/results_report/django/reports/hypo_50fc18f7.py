#!/usr/bin/env python3
"""
Hypothesis test for Django CSRF trusted origins validation bug.
This test verifies that origins must have a scheme at the beginning.
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

from hypothesis import given, strategies as st, example
from unittest.mock import patch
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

@given(st.text())
@example("://example.com")  # Explicit failing case
def test_scheme_must_be_at_start(origin):
    """
    Test that CSRF_TRUSTED_ORIGINS validation properly checks for scheme at start.

    The Django documentation states that origins must start with a scheme,
    and the error message says "must start with a scheme", but the actual
    validation only checks if "://" exists anywhere in the string.
    """
    with patch('django.conf.settings.CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(app_configs=None)

        if '://' in origin:
            index_of_separator = origin.index('://')
            if index_of_separator == 0:
                # If :// is at the start, there's no scheme before it
                assert len(errors) > 0, \
                    f"Origin '{origin}' has no scheme before ://, should fail"
        else:
            # If there's no :// at all, it should fail
            assert len(errors) > 0, \
                f"Origin '{origin}' has no ://, should fail"

if __name__ == "__main__":
    # Run the test with hypothesis
    test_scheme_must_be_at_start()