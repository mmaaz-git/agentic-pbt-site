#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings, assume
from django.apps.config import AppConfig

failures = []

@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz.', min_size=1, max_size=30))
@settings(max_examples=1000)
def test_create_rpartition_edge_cases(entry):
    global failures
    mod_path, _, cls_name = entry.rpartition(".")

    if mod_path and not cls_name:
        try:
            config = AppConfig.create(entry)
        except IndexError as e:
            failures.append((entry, str(e)))
            assert False, f"IndexError should not occur for entry={repr(entry)}: {e}"
        except Exception:
            # Other exceptions are fine
            pass

# Run the test
print("Running hypothesis test...")
try:
    test_create_rpartition_edge_cases()
    print("Test passed all examples!")
except AssertionError as e:
    print(f"Test failed: {e}")
    print(f"\nFound {len(failures)} failing inputs that cause IndexError:")
    for entry, error in failures[:10]:  # Show first 10
        print(f"  {repr(entry)} -> {error}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")