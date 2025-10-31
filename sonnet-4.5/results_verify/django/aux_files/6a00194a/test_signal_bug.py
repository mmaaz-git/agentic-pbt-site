#!/usr/bin/env python
import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.10/site-packages')

from django.dispatch import Signal

print("Testing Django Signal with use_caching=True and non-weakly-referenceable senders...")
print("=" * 70)

# Test 1: sender=None
print("\nTest 1: sender=None")
print("-" * 40)
try:
    signal = Signal(use_caching=True)

    def my_receiver(signal, sender, **kwargs):
        return "received"

    signal.connect(my_receiver, sender=None, weak=False)
    result = signal.send(sender=None)
    print("✓ Success! Result:", result)
except TypeError as e:
    print(f"✗ Failed with TypeError: {e}")
except Exception as e:
    print(f"✗ Failed with {type(e).__name__}: {e}")

# Test 2: sender=object()
print("\nTest 2: sender=object()")
print("-" * 40)
try:
    signal = Signal(use_caching=True)
    sender = object()

    def my_receiver(signal, sender, **kwargs):
        return "received"

    signal.connect(my_receiver, sender=sender, weak=False)
    result = signal.send(sender=sender)
    print("✓ Success! Result:", result)
except TypeError as e:
    print(f"✗ Failed with TypeError: {e}")
except Exception as e:
    print(f"✗ Failed with {type(e).__name__}: {e}")

# Test 3: sender=42 (int)
print("\nTest 3: sender=42 (int)")
print("-" * 40)
try:
    signal = Signal(use_caching=True)

    def my_receiver(signal, sender, **kwargs):
        return "received"

    signal.connect(my_receiver, sender=42, weak=False)
    result = signal.send(sender=42)
    print("✓ Success! Result:", result)
except TypeError as e:
    print(f"✗ Failed with TypeError: {e}")
except Exception as e:
    print(f"✗ Failed with {type(e).__name__}: {e}")

# Test 4: sender="string"
print("\nTest 4: sender='string'")
print("-" * 40)
try:
    signal = Signal(use_caching=True)

    def my_receiver(signal, sender, **kwargs):
        return "received"

    signal.connect(my_receiver, sender="string", weak=False)
    result = signal.send(sender="string")
    print("✓ Success! Result:", result)
except TypeError as e:
    print(f"✗ Failed with TypeError: {e}")
except Exception as e:
    print(f"✗ Failed with {type(e).__name__}: {e}")

# Test 5: sender=custom class instance (should work)
print("\nTest 5: sender=custom class instance")
print("-" * 40)
try:
    signal = Signal(use_caching=True)

    class MyClass:
        pass

    sender = MyClass()

    def my_receiver(signal, sender, **kwargs):
        return "received"

    signal.connect(my_receiver, sender=sender, weak=False)
    result = signal.send(sender=sender)
    print("✓ Success! Result:", result)
except TypeError as e:
    print(f"✗ Failed with TypeError: {e}")
except Exception as e:
    print(f"✗ Failed with {type(e).__name__}: {e}")

# Test 6: Same tests with use_caching=False (should all work)
print("\n" + "=" * 70)
print("Testing with use_caching=False (control test)...")
print("=" * 70)

print("\nTest 6: use_caching=False with sender=None")
print("-" * 40)
try:
    signal = Signal(use_caching=False)

    def my_receiver(signal, sender, **kwargs):
        return "received"

    signal.connect(my_receiver, sender=None, weak=False)
    result = signal.send(sender=None)
    print("✓ Success! Result:", result)
except Exception as e:
    print(f"✗ Failed with {type(e).__name__}: {e}")

print("\nTest 7: use_caching=False with sender=42")
print("-" * 40)
try:
    signal = Signal(use_caching=False)

    def my_receiver(signal, sender, **kwargs):
        return "received"

    signal.connect(my_receiver, sender=42, weak=False)
    result = signal.send(sender=42)
    print("✓ Success! Result:", result)
except Exception as e:
    print(f"✗ Failed with {type(e).__name__}: {e}")