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


# First, let's run the Hypothesis test
print("Running Hypothesis test...")
try:
    @given(st.text(min_size=1))
    def test_signal_with_caching_and_string_sender(sender):
        signal = Signal(use_caching=True)
        signal.connect(receiver, sender=sender, weak=False)
        responses = signal.send(sender=sender)
        assert len(responses) > 0

    # Run the test with a simple input
    test_signal_with_caching_and_string_sender('0')
    print("Hypothesis test PASSED")
except Exception as e:
    print(f"Hypothesis test FAILED: {e}")
    traceback.print_exc()


# Now let's run the direct reproduction case
print("\nRunning direct reproduction case...")
try:
    signal = Signal(use_caching=True)
    sender = "my_sender"

    signal.connect(receiver, sender=sender, weak=False)
    print(f"Connected receiver to signal with sender={repr(sender)}")

    responses = signal.send(sender=sender)
    print(f"Send succeeded! Responses: {responses}")
except TypeError as e:
    print(f"Error occurred as expected: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()


# Test with use_caching=False to show it works
print("\nTesting with use_caching=False...")
try:
    signal_no_cache = Signal(use_caching=False)
    sender = "my_sender"

    signal_no_cache.connect(receiver, sender=sender, weak=False)
    print(f"Connected receiver to signal (no caching) with sender={repr(sender)}")

    responses = signal_no_cache.send(sender=sender)
    print(f"Send succeeded! Responses: {responses}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()


# Test with a weakly-referenceable object
print("\nTesting with weakly-referenceable object (class instance)...")
try:
    class MySender:
        pass

    signal_cache = Signal(use_caching=True)
    sender = MySender()

    signal_cache.connect(receiver, sender=sender, weak=False)
    print(f"Connected receiver to signal with sender={sender}")

    responses = signal_cache.send(sender=sender)
    print(f"Send succeeded! Responses: {responses}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()