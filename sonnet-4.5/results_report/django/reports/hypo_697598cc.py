#!/usr/bin/env python3
"""Property-based test for Django AppConfig.create() trailing dot bug."""

import os
import sys
import django
from django.conf import settings
from hypothesis import given, strategies as st, settings as hypo_settings, HealthCheck, seed
import traceback

# Configure minimal Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': ':memory:',
            }
        }
    )

# Setup Django
django.setup()

from django.apps.config import AppConfig

@given(st.text(min_size=1).filter(lambda s: '.' in s))
@hypo_settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
@seed(54184995177019238848460567174500504572)
def test_create_with_module_paths(entry):
    """Test that AppConfig.create() handles various module path formats gracefully."""
    if entry.endswith('.') and not entry.rpartition('.')[2]:
        # This should trigger an IndexError due to the bug
        try:
            AppConfig.create(entry)
            # If we get here without exception, that's unexpected but not a failure for this specific bug
            pass
        except IndexError as e:
            # This is the bug - we get IndexError instead of a proper error
            print(f"Found IndexError for input: {repr(entry)}")
            print(f"  Error message: {e}")
            # Re-raise to let Hypothesis know this is a failure
            raise
        except (ImportError, Exception):
            # Other exceptions are expected for invalid module paths
            pass
    else:
        # For non-trailing-dot entries, we expect either success or a proper exception
        try:
            AppConfig.create(entry)
        except IndexError:
            # IndexError should not happen for non-trailing-dot entries
            print(f"Unexpected IndexError for input: {repr(entry)}")
            raise
        except (ImportError, Exception):
            # Other exceptions are fine
            pass

if __name__ == "__main__":
    print("Running property-based test for AppConfig.create() trailing dot bug...")
    print("=" * 60)

    try:
        test_create_with_module_paths()
        print("Test completed without finding the bug (unexpected)")
    except Exception as e:
        print("\nTest failed (bug found)!")
        print("-" * 60)
        print("Full traceback:")
        traceback.print_exc()