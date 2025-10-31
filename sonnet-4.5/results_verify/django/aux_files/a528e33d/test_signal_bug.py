#!/usr/bin/env python3
"""Test script to reproduce the Signal caching bug with non-weakrefable senders."""

import sys
import traceback
from hypothesis import given, strategies as st
from django.dispatch import Signal


def test_hypothesis():
    """Run the hypothesis test from the bug report."""
    print("Running Hypothesis test...")

    @given(st.booleans())
    def test_connect_disconnect_roundtrip(use_caching):
        signal = Signal(use_caching=use_caching)

        def receiver(sender, **kwargs):
            return "received"

        signal.connect(receiver)
        assert signal.has_listeners()

        result = signal.disconnect(receiver)
        assert result == True
        assert not signal.has_listeners()

    try:
        test_connect_disconnect_roundtrip()
        print("Hypothesis test passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")
        traceback.print_exc()
        return False
    return True


def test_manual_none_sender():
    """Test sending signal with None sender when use_caching=True."""
    print("\nTesting with sender=None and use_caching=True...")

    signal = Signal(use_caching=True)

    def receiver(sender, **kwargs):
        return "received"

    signal.connect(receiver)

    try:
        result = signal.send(sender=None)
        print(f"Signal sent successfully with sender=None: {result}")
        return True
    except TypeError as e:
        print(f"TypeError occurred: {e}")
        traceback.print_exc()
        return False


def test_manual_object_sender():
    """Test sending signal with plain object() sender when use_caching=True."""
    print("\nTesting with sender=object() and use_caching=True...")

    signal = Signal(use_caching=True)

    def receiver(sender, **kwargs):
        return "received"

    signal.connect(receiver)

    try:
        result = signal.send(sender=object())
        print(f"Signal sent successfully with sender=object(): {result}")
        return True
    except TypeError as e:
        print(f"TypeError occurred: {e}")
        traceback.print_exc()
        return False


def test_without_caching():
    """Test that everything works fine when use_caching=False."""
    print("\nTesting with use_caching=False (control test)...")

    signal = Signal(use_caching=False)

    def receiver(sender, **kwargs):
        return "received"

    signal.connect(receiver)

    try:
        # Test with None
        result1 = signal.send(sender=None)
        print(f"Signal sent successfully with sender=None (no caching): {result1}")

        # Test with object()
        result2 = signal.send(sender=object())
        print(f"Signal sent successfully with sender=object() (no caching): {result2}")

        return True
    except Exception as e:
        print(f"Unexpected error without caching: {e}")
        traceback.print_exc()
        return False


def test_has_listeners():
    """Test has_listeners() method with use_caching=True."""
    print("\nTesting has_listeners() with use_caching=True...")

    signal = Signal(use_caching=True)

    def receiver(sender, **kwargs):
        return "received"

    signal.connect(receiver)

    try:
        # This might also trigger the bug according to the report
        has_listeners = signal.has_listeners()
        print(f"has_listeners() returned: {has_listeners}")
        return True
    except TypeError as e:
        print(f"TypeError in has_listeners(): {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Django Signal Caching Bug Reproduction")
    print("=" * 60)

    results = {
        "hypothesis_test": test_hypothesis(),
        "manual_none_sender": test_manual_none_sender(),
        "manual_object_sender": test_manual_object_sender(),
        "without_caching": test_without_caching(),
        "has_listeners": test_has_listeners()
    }

    print("\n" + "=" * 60)
    print("Summary of Results:")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")

    print("\nConclusion:")
    if not results["manual_none_sender"] and not results["manual_object_sender"]:
        print("BUG CONFIRMED: Signal with use_caching=True crashes with non-weakrefable senders")
    else:
        print("BUG NOT REPRODUCED: Signal works correctly with all sender types")