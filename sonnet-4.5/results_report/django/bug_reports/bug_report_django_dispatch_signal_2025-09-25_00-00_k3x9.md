# Bug Report: django.dispatch.Signal WeakKeyDictionary with None Sender

**Target**: `django.dispatch.Signal`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `use_caching=True` is enabled on a Signal, calling any method that checks the cache with `sender=None` raises a `TypeError` because `WeakKeyDictionary` cannot use `None` as a key.

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
```

**Failing input**: `use_caching=True`

## Reproducing the Bug

```python
from django.dispatch import Signal

signal = Signal(use_caching=True)

def receiver(sender, **kwargs):
    return "response"

signal.connect(receiver, weak=False)
responses = signal.send(sender=None)
```

Output:
```
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    responses = signal.send(sender=None)
  File ".../django/dispatch/dispatcher.py", line 183, in send
    or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
  File ".../weakref.py", line 452, in get
    return self.data.get(ref(key),default)
TypeError: cannot create weak reference to 'NoneType' object
```

## Why This Is A Bug

The Signal API allows `sender=None` as a valid value (meaning "any sender" or "no specific sender"). This is documented in the `connect()` method:

> sender: The sender to which the receiver should respond. Must either be a Python object, or None to receive events from any sender.

However, when `use_caching=True`, the implementation uses a `WeakKeyDictionary` to cache receivers by sender:

```python
self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
```

Since `WeakKeyDictionary` cannot create weak references to `None`, any operation that tries to use `None` as a cache key crashes. This affects:
- `Signal.send(sender=None)`
- `Signal.has_listeners(sender=None)`
- `Signal._live_receivers(sender=None)`

This violates the documented API contract that `sender=None` is valid.

## Fix

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -180,7 +180,9 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (
+                sender is not None and self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            )
         ):
             return []
         responses = []
@@ -229,7 +231,9 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (
+                sender is not None and self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            )
         ):
             return []
         sync_receivers, async_receivers = self._live_receivers(sender)
@@ -293,7 +297,9 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (
+                sender is not None and self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            )
         ):
             return []

@@ -359,7 +365,9 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (
+                sender is not None and self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            )
         ):
             return []

@@ -422,7 +430,7 @@ class Signal:
         """
         receivers = None
         if self.use_caching and not self._dead_receivers:
-            receivers = self.sender_receivers_cache.get(sender)
+            receivers = None if sender is None else self.sender_receivers_cache.get(sender)
             # We could end up here with NO_RECEIVERS even if we do check this case in
             # .send() prior to calling _live_receivers() due to concurrent .send() call.
             if receivers is NO_RECEIVERS:
@@ -437,7 +445,7 @@ class Signal:
                         receivers.append((receiver, is_async))
                 if self.use_caching:
                     if not receivers:
-                        self.sender_receivers_cache[sender] = NO_RECEIVERS
+                        if sender is not None: self.sender_receivers_cache[sender] = NO_RECEIVERS
                     else:
                         # Note, we must cache the weakref versions.
-                        self.sender_receivers_cache[sender] = receivers
+                        if sender is not None: self.sender_receivers_cache[sender] = receivers
```