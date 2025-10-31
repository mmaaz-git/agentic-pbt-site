# Bug Report: django.dispatch.Signal Crashes with Non-Weakrefable Senders When Caching Enabled

**Target**: `django.dispatch.Signal`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Django's Signal class crashes with a TypeError when `use_caching=True` is used with senders that cannot be weakly referenced, including the documented `sender=None` pattern and plain object instances.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.dispatch import Signal


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
```

<details>

<summary>
**Failing input**: `use_caching=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 21, in <module>
    test_connect_disconnect_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_connect_disconnect_roundtrip
    def test_connect_disconnect_roundtrip(use_caching):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 13, in test_connect_disconnect_roundtrip
    assert signal.has_listeners()
           ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/dispatch/dispatcher.py", line 156, in has_listeners
    sync_receivers, async_receivers = self._live_receivers(sender)
                                      ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/dispatch/dispatcher.py", line 425, in _live_receivers
    receivers = self.sender_receivers_cache.get(sender)
  File "/home/npc/miniconda/lib/python3.13/weakref.py", line 452, in get
    return self.data.get(ref(key),default)
                         ~~~^^^^^
TypeError: cannot create weak reference to 'NoneType' object
Falsifying example: test_connect_disconnect_roundtrip(
    use_caching=True,
)
```
</details>

## Reproducing the Bug

```python
from django.dispatch import Signal

# Create a signal with caching enabled
signal = Signal(use_caching=True)

# Define a simple receiver function
def receiver(sender, **kwargs):
    return "received"

# Connect the receiver to the signal
signal.connect(receiver)

# Try to send a signal with sender=None (documented as valid)
# This should work but will crash due to the WeakKeyDictionary
print("Attempting to send signal with sender=None...")
try:
    result = signal.send(sender=None)
    print(f"Success: {result}")
except TypeError as e:
    print(f"Error: {e}")

# Also try with a plain object instance
print("\nAttempting to send signal with sender=object()...")
try:
    result = signal.send(sender=object())
    print(f"Success: {result}")
except TypeError as e:
    print(f"Error: {e}")

# Show that has_listeners() also crashes
print("\nAttempting to check has_listeners()...")
try:
    has_listeners = signal.has_listeners()
    print(f"Has listeners: {has_listeners}")
except TypeError as e:
    print(f"Error: {e}")

# Demonstrate that it works fine without caching
print("\n--- Testing without caching ---")
signal_no_cache = Signal(use_caching=False)
signal_no_cache.connect(receiver)

print("Sending with sender=None (no caching)...")
result = signal_no_cache.send(sender=None)
print(f"Success: {result}")

print("Sending with sender=object() (no caching)...")
result = signal_no_cache.send(sender=object())
print(f"Success: {result}")
```

<details>

<summary>
TypeError crashes when attempting to use non-weakrefable senders with caching enabled
</summary>
```
Attempting to send signal with sender=None...
Error: cannot create weak reference to 'NoneType' object

Attempting to send signal with sender=object()...
Error: cannot create weak reference to 'object' object

Attempting to check has_listeners()...
Error: cannot create weak reference to 'NoneType' object

--- Testing without caching ---
Sending with sender=None (no caching)...
Success: [(<function receiver at 0x78d5303a34c0>, 'received')]
Sending with sender=object() (no caching)...
Success: [(<function receiver at 0x78d5303a34c0>, 'received')]
```
</details>

## Why This Is A Bug

This bug violates Django's documented Signal behavior and the principle of transparent optimization. The Signal class documentation explicitly states in the source code (lines 69-71 of dispatcher.py) that the sender parameter "Must either be a Python object, or None to receive events from any sender." The `sender=None` pattern is a core Django feature used to create receivers that respond to signals from any sender.

The `use_caching` parameter is intended as a performance optimization that should be transparent to users. However, when enabled, it fundamentally breaks the Signal API by restricting which types of objects can be used as senders. The implementation uses `weakref.WeakKeyDictionary()` for the cache (line 47), which requires all keys to support weak references. Many common Python types cannot be weakly referenced:

- `None` (NoneType) - explicitly documented as a valid sender
- Plain `object()` instances without `__weakref__` slots
- Built-in immutable types like integers, strings, floats, tuples

This creates a silent breaking change where code that works perfectly with `use_caching=False` crashes with cryptic TypeErrors when `use_caching=True`. The error message "cannot create weak reference to 'NoneType' object" provides no guidance about the root cause or how to fix it. Multiple Signal methods are affected: `send()`, `has_listeners()`, `send_robust()`, `asend()`, and `asend_robust()`, all of which call `_live_receivers()` which attempts to access the WeakKeyDictionary cache.

## Relevant Context

The bug occurs in multiple locations in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/dispatch/dispatcher.py`:

- Line 47: Cache initialization using `weakref.WeakKeyDictionary()`
- Line 156: `has_listeners()` calls `_live_receivers(sender)` with default `sender=None`
- Line 183: `send()` method tries to access cache with `.get(sender)`
- Line 425: `_live_receivers()` tries to access cache, triggering the weak reference creation

Django's internal usage pattern shows that `sender=None` is fundamental to the framework's signal system. Many Django applications rely on universal receivers that listen to signals from any sender. The `use_caching` optimization should not break this core functionality.

Documentation references:
- Django Signals Documentation: https://docs.djangoproject.com/en/stable/topics/signals/
- Source code: https://github.com/django/django/blob/main/django/dispatch/dispatcher.py

## Proposed Fix

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -44,7 +44,16 @@ class Signal:
         # distinct sender we cache the receivers that sender has in
         # 'sender_receivers_cache'. The cache is cleaned when .connect() or
         # .disconnect() is called and populated on send().
-        self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
+        if use_caching:
+            # Use a regular dict to support non-weakrefable keys like None
+            # The cache is cleared on connect/disconnect operations to prevent
+            # memory leaks, so using a regular dict is safe
+            self.sender_receivers_cache = {}
+        else:
+            self.sender_receivers_cache = {}
         self._dead_receivers = False

     def connect(self, receiver, sender=None, weak=True, dispatch_uid=None):
```