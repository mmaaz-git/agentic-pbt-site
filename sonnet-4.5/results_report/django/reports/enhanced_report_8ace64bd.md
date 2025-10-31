# Bug Report: django.dispatch.Signal TypeError with None Sender and Caching

**Target**: `django.dispatch.Signal`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `use_caching=True` is enabled on a Django Signal, calling any signal method with `sender=None` raises a `TypeError` because `WeakKeyDictionary` cannot create weak references to `None`.

## Property-Based Test

```python
from django.dispatch import Signal
from hypothesis import given, strategies as st

@given(st.booleans())
def test_send_with_none_sender(use_caching):
    signal = Signal(use_caching=use_caching)

    def receiver(sender, **kwargs):
        return "response"

    signal.connect(receiver, weak=False)
    responses = signal.send(sender=None)
    assert len(responses) == 1

# Run the test
if __name__ == "__main__":
    test_send_with_none_sender()
```

<details>

<summary>
**Failing input**: `use_caching=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 17, in <module>
    test_send_with_none_sender()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 5, in test_send_with_none_sender
    def test_send_with_none_sender(use_caching):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 12, in test_send_with_none_sender
    responses = signal.send(sender=None)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/dispatch/dispatcher.py", line 183, in send
    or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/weakref.py", line 452, in get
    return self.data.get(ref(key),default)
                         ~~~^^^^^
TypeError: cannot create weak reference to 'NoneType' object
Falsifying example: test_send_with_none_sender(
    use_caching=True,
)
```
</details>

## Reproducing the Bug

```python
from django.dispatch import Signal

# Create a signal with caching enabled
signal = Signal(use_caching=True)

# Define a receiver function
def receiver(sender, **kwargs):
    return "response"

# Connect the receiver
signal.connect(receiver, weak=False)

# Try to send with sender=None
print("Attempting to send signal with sender=None...")
responses = signal.send(sender=None)
print(f"Responses: {responses}")
```

<details>

<summary>
TypeError when sending signal with None sender and caching enabled
</summary>
```
Attempting to send signal with sender=None...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/repo.py", line 15, in <module>
    responses = signal.send(sender=None)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/dispatch/dispatcher.py", line 183, in send
    or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/weakref.py", line 452, in get
    return self.data.get(ref(key),default)
                         ~~~^^^^^
TypeError: cannot create weak reference to 'NoneType' object
```
</details>

## Why This Is A Bug

The Django Signal API explicitly documents that `sender=None` is a valid value, as stated in the `connect()` method documentation (dispatcher.py:69-71):

> "sender: The sender to which the receiver should respond. Must either be a Python object, or None to receive events from any sender."

The implementation even includes special handling for `None` senders with the `NONE_ID` constant (dispatcher.py:19) and checks for it in `_live_receivers()` (dispatcher.py:436). However, when `use_caching=True`, the code uses `weakref.WeakKeyDictionary()` which cannot handle `None` as a key because Python's weak references cannot be created for `NoneType`.

This bug causes crashes in multiple critical methods:
- `Signal.send(sender=None)`
- `Signal.send_robust(sender=None)`
- `Signal.has_listeners(sender=None)`
- `Signal.asend(sender=None)`
- `Signal.asend_robust(sender=None)`

The crash occurs at lines 183, 232, 296, and 361 in dispatcher.py where `self.sender_receivers_cache.get(sender)` is called, and at line 425 in `_live_receivers()`. This violates the documented API contract that explicitly allows `sender=None`.

## Relevant Context

The Django signals system is widely used for decoupling components in Django applications. The ability to use `sender=None` is important for creating generic receivers that respond to signals from any sender. The `use_caching` parameter is an optimization feature that improves performance by caching receiver lookups per sender.

The bug occurs because Python's `weakref.WeakKeyDictionary` fundamentally cannot create weak references to `None`, `int`, `str`, and other built-in immutable types. The Django code needs to handle this special case explicitly when caching is enabled.

Django documentation: https://docs.djangoproject.com/en/stable/topics/signals/
Source code: https://github.com/django/django/blob/main/django/dispatch/dispatcher.py

## Proposed Fix

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -180,7 +180,7 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (sender is not None and self.sender_receivers_cache.get(sender) is NO_RECEIVERS)
         ):
             return []
         responses = []
@@ -230,7 +230,7 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (sender is not None and self.sender_receivers_cache.get(sender) is NO_RECEIVERS)
         ):
             return []
         sync_receivers, async_receivers = self._live_receivers(sender)
@@ -294,7 +294,7 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (sender is not None and self.sender_receivers_cache.get(sender) is NO_RECEIVERS)
         ):
             return []

@@ -359,7 +359,7 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (sender is not None and self.sender_receivers_cache.get(sender) is NO_RECEIVERS)
         ):
             return []

@@ -423,7 +423,7 @@ class Signal:
         """
         receivers = None
         if self.use_caching and not self._dead_receivers:
-            receivers = self.sender_receivers_cache.get(sender)
+            receivers = self.sender_receivers_cache.get(sender) if sender is not None else None
             # We could end up here with NO_RECEIVERS even if we do check this case in
             # .send() prior to calling _live_receivers() due to concurrent .send() call.
             if receivers is NO_RECEIVERS:
@@ -438,10 +438,12 @@ class Signal:
                         receivers.append((receiver, is_async))
                 if self.use_caching:
                     if not receivers:
-                        self.sender_receivers_cache[sender] = NO_RECEIVERS
+                        if sender is not None:
+                            self.sender_receivers_cache[sender] = NO_RECEIVERS
                     else:
                         # Note, we must cache the weakref versions.
-                        self.sender_receivers_cache[sender] = receivers
+                        if sender is not None:
+                            self.sender_receivers_cache[sender] = receivers
         non_weak_sync_receivers = []
         non_weak_async_receivers = []
         for receiver, is_async in receivers:
```