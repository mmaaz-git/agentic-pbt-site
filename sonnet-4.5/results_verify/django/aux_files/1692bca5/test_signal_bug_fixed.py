#!/usr/bin/env python
"""Test to reproduce the Django Signal bug with use_caching=True"""

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=False)

from hypothesis import given, strategies as st
from django.dispatch import Signal
import traceback


def receiver(**kwargs):
    return "received"


# First, let's run the Hypothesis test properly
print("Running Hypothesis test...")
@given(st.text(min_size=1))
def test_signal_with_caching_and_string_sender(sender):
    signal = Signal(use_caching=True)
    signal.connect(receiver, sender=sender, weak=False)
    responses = signal.send(sender=sender)
    assert len(responses) > 0

try:
    # Run the test with hypothesis
    test_signal_with_caching_and_string_sender()
    print("Hypothesis test PASSED")
except Exception as e:
    print(f"Hypothesis test FAILED with error: {e}")
    print("This confirms the bug - string senders cannot be used with use_caching=True")


# Now let's run the direct reproduction case
print("\nDirect reproduction case with string sender:")
print("=" * 50)
try:
    signal = Signal(use_caching=True)
    sender = "my_sender"

    signal.connect(receiver, sender=sender, weak=False)
    print(f"✓ Connected receiver to signal with sender={repr(sender)}")

    responses = signal.send(sender=sender)
    print(f"✓ Send succeeded! Responses: {responses}")
except TypeError as e:
    print(f"✗ Error occurred: {e}")
    print("  This is the bug - string senders fail with use_caching=True")


# Test with use_caching=False to show it works
print("\nTesting with use_caching=False:")
print("=" * 50)
try:
    signal_no_cache = Signal(use_caching=False)
    sender = "my_sender"

    signal_no_cache.connect(receiver, sender=sender, weak=False)
    print(f"✓ Connected receiver to signal (no caching) with sender={repr(sender)}")

    responses = signal_no_cache.send(sender=sender)
    print(f"✓ Send succeeded! Responses: {responses}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")


# Test with a weakly-referenceable object
print("\nTesting with weakly-referenceable object (class instance):")
print("=" * 50)
try:
    class MySender:
        pass

    signal_cache = Signal(use_caching=True)
    sender = MySender()

    signal_cache.connect(receiver, sender=sender, weak=False)
    print(f"✓ Connected receiver to signal with sender={sender}")

    responses = signal_cache.send(sender=sender)
    print(f"✓ Send succeeded! Responses: {responses}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")


# Test with other non-weakly-referenceable types
print("\nTesting with other non-weakly-referenceable types:")
print("=" * 50)

test_cases = [
    ("Integer", 42),
    ("Float", 3.14),
    ("None", None),
    ("Tuple", (1, 2, 3)),
    ("Boolean", True),
]

for name, sender in test_cases:
    try:
        signal = Signal(use_caching=True)
        signal.connect(receiver, sender=sender, weak=False)
        responses = signal.send(sender=sender)
        print(f"✓ {name} sender ({repr(sender)}): SUCCESS")
    except TypeError as e:
        print(f"✗ {name} sender ({repr(sender)}): FAILED - {e}")