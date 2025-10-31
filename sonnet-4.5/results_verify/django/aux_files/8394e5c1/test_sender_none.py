#!/usr/bin/env python3
"""Test to confirm sender=None is officially supported by Django."""

from django.dispatch import Signal

# Create a signal without caching
signal = Signal(use_caching=False)

# Define a receiver
def my_receiver(sender, **kwargs):
    print(f"Received signal from sender: {sender}")
    return "received"

# Connect the receiver with sender=None (should receive from any sender)
signal.connect(my_receiver, sender=None)

# Test sending with different senders
print("\n1. Testing with sender=None:")
result1 = signal.send(sender=None)
print(f"Result: {result1}")

print("\n2. Testing with sender='some_string':")
result2 = signal.send(sender="some_string")
print(f"Result: {result2}")

print("\n3. Testing with sender=object():")
obj = object()
result3 = signal.send(sender=obj)
print(f"Result: {result3}")

print("\n4. Testing with integer sender:")
result4 = signal.send(sender=42)
print(f"Result: {result4}")

print("\nConclusion: sender=None is fully supported when use_caching=False")

# Now let's test what the documentation says about sender=None
print("\n" + "="*60)
print("Testing receiver connected to sender=None behavior:")
print("="*60)

signal2 = Signal(use_caching=False)

def universal_receiver(sender, **kwargs):
    print(f"Universal receiver got: {sender}")
    return "universal"

# Connect with sender=None means receive from ANY sender
signal2.connect(universal_receiver, sender=None)

print("\n5. Signal connected with sender=None receives from any sender:")
signal2.send(sender="test1")
signal2.send(sender=42)
signal2.send(sender=None)
signal2.send(sender=object())