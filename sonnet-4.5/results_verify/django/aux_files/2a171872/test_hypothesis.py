#!/usr/bin/env python3
import os
import sys

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
import django
from django.conf import settings
settings.configure(DEBUG=True)

from hypothesis import given, settings as hyp_settings, strategies as st
from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation

print("=== Running Hypothesis Test ===")
print()

# Track failures
failures = []

@given(st.text(alphabet=st.characters(blacklist_categories=['Cc', 'Cs']), min_size=1, max_size=100))
@hyp_settings(max_examples=1000)
def test_validate_rejects_backslash_as_separator(name):
    if '\\' in name and os.path.basename(name) not in {"", ".", ".."}:
        try:
            validate_file_name(name, allow_relative_path=False)
            # According to the bug report, this should fail
            failures.append(name)
            assert False, f"Should reject backslash in filename: {name!r}"
        except SuspiciousFileOperation:
            # This is expected according to the bug report
            pass

try:
    test_validate_rejects_backslash_as_separator()
    print("Test completed, but no assertion errors were caught by hypothesis")
except AssertionError as e:
    print(f"Hypothesis test failed as expected: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

if failures:
    print(f"\nFound {len(failures)} cases where backslash was accepted:")
    for i, fail in enumerate(failures[:10]):  # Show first 10
        print(f"  {i+1}. {fail!r}")
    if len(failures) > 10:
        print(f"  ... and {len(failures)-10} more")
else:
    print("\nNo failures found - all inputs with backslashes were rejected")