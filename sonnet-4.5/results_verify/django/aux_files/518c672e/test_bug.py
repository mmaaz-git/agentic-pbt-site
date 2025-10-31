#!/usr/bin/env python3
"""Test the reported bug in django.dispatch.Signal"""

# First, let's test the hypothesis test
from hypothesis import given, strategies as st
from django.dispatch import Signal

@given(st.booleans())
def test_send_with_none_sender(use_caching):
    signal = Signal(use_caching=use_caching)

    def receiver(sender, **kwargs):
        return "response"

    signal.connect(receiver)
    responses = signal.send(sender=None)

    assert isinstance(responses, list)

# Run the hypothesis test
print("Running hypothesis test...")
try:
    test_send_with_none_sender()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Now test the manual reproduction
print("\nManual reproduction test...")
print("Testing with use_caching=False...")
try:
    signal = Signal(use_caching=False)
    def my_receiver(sender, **kwargs):
        return "response"
    signal.connect(my_receiver)
    responses = signal.send(sender=None)
    print(f"Success with use_caching=False: {responses}")
except Exception as e:
    print(f"Failed with use_caching=False: {e}")

print("\nTesting with use_caching=True...")
try:
    signal = Signal(use_caching=True)
    def my_receiver(sender, **kwargs):
        return "response"
    signal.connect(my_receiver)
    responses = signal.send(sender=None)
    print(f"Success with use_caching=True: {responses}")
except Exception as e:
    print(f"Failed with use_caching=True: {type(e).__name__}: {e}")

# Test with string sender
print("\nTesting with string sender and use_caching=True...")
try:
    signal = Signal(use_caching=True)
    def my_receiver(sender, **kwargs):
        return "response"
    signal.connect(my_receiver)
    responses = signal.send(sender="my_string_sender")
    print(f"Success with string sender: {responses}")
except Exception as e:
    print(f"Failed with string sender: {type(e).__name__}: {e}")

# Test with int sender
print("\nTesting with int sender and use_caching=True...")
try:
    signal = Signal(use_caching=True)
    def my_receiver(sender, **kwargs):
        return "response"
    signal.connect(my_receiver)
    responses = signal.send(sender=42)
    print(f"Success with int sender: {responses}")
except Exception as e:
    print(f"Failed with int sender: {type(e).__name__}: {e}")

# Test with object instance sender (should work)
print("\nTesting with object instance sender and use_caching=True...")
try:
    signal = Signal(use_caching=True)
    def my_receiver(sender, **kwargs):
        return "response"
    signal.connect(my_receiver)

    class MyClass:
        pass

    obj = MyClass()
    responses = signal.send(sender=obj)
    print(f"Success with object instance: {responses}")
except Exception as e:
    print(f"Failed with object instance: {type(e).__name__}: {e}")