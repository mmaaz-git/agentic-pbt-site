# Bug Report: django.dispatch Signal WeakKeyDictionary TypeError

**Target**: `django.dispatch.Signal`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When a Signal is created with `use_caching=True`, calling `send()` with a non-weakly-referenceable sender (such as `None`, `object()`, `int`, or `str`) causes a `TypeError: cannot create weak reference` crash. This affects Django's own model signals which use caching.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.dispatch import Signal


@settings(max_examples=500)
@given(st.booleans())
def test_cache_clearing_on_connect(use_caching):
    signal = Signal(use_caching=use_caching)
    sender = object()

    def receiver1(**kwargs):
        return 1

    signal.connect(receiver1, sender=sender, weak=False)
    signal.send(sender)

    def receiver2(**kwargs):
        return 2

    signal.connect(receiver2, sender=sender, weak=False)
    responses = signal.send(sender)

    assert len(responses) == 2
```

**Failing input**: `use_caching=True`

## Reproducing the Bug

```python
from django.dispatch import Signal

signal = Signal(use_caching=True)

def my_receiver(signal, sender, **kwargs):
    return "received"

signal.connect(my_receiver, sender=None, weak=False)
signal.send(sender=None)
```

**Output:**
```
TypeError: cannot create weak reference to 'NoneType' object
```

This also fails with:
- `sender=object()`
- `sender=42`
- `sender="string"`

But works with custom class instances that support weak references.

## Why This Is A Bug

1. **API contract violation**: The `Signal` class accepts `use_caching=True` as a parameter and any object as a sender, but crashes with common sender types when both are combined.

2. **No documentation of restrictions**: There is no warning that enabling caching restricts the types of senders that can be used.

3. **Affects Django's own code**: Django's model signals (pre_save, post_save, etc.) use `use_caching=True`. While model classes are typically weakly referenceable, users can legally pass `sender=None` to receive signals from all senders, which would crash.

4. **Violates principle of least surprise**: Users wouldn't expect that enabling a performance optimization would restrict the types of valid inputs.

## Fix

The issue is at line 47 in `dispatcher.py`, where a `WeakKeyDictionary` is used without handling non-weakly-referenceable keys. The fix should gracefully handle these cases by catching `TypeError` when accessing the cache:

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -180,7 +180,12 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (
+                self.use_caching
+                and self._get_cached_receivers(sender) is NO_RECEIVERS
+            )
+            or (
+                not self.use_caching
+                and self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            )
         ):
             return []
         responses = []
@@ -422,7 +427,14 @@ class Signal:
         """
         receivers = None
         if self.use_caching and not self._dead_receivers:
-            receivers = self.sender_receivers_cache.get(sender)
+            receivers = self._get_cached_receivers(sender)
             # We could end up here with NO_RECEIVERS even if we do check this case in
             # .send() prior to calling _live_receivers() due to concurrent .send() call.
             if receivers is NO_RECEIVERS:
@@ -437,10 +449,19 @@ class Signal:
                         receivers.append((receiver, is_async))
                 if self.use_caching:
                     if not receivers:
-                        self.sender_receivers_cache[sender] = NO_RECEIVERS
+                        self._set_cached_receivers(sender, NO_RECEIVERS)
                     else:
                         # Note, we must cache the weakref versions.
-                        self.sender_receivers_cache[sender] = receivers
+                        self._set_cached_receivers(sender, receivers)
+
+    def _get_cached_receivers(self, sender):
+        try:
+            return self.sender_receivers_cache.get(sender)
+        except TypeError:
+            return None
+
+    def _set_cached_receivers(self, sender, value):
+        try:
+            self.sender_receivers_cache[sender] = value
+        except TypeError:
+            pass
```

This fix allows the cache to silently skip non-weakly-referenceable senders, falling back to the non-cached code path for those cases. This maintains backward compatibility while fixing the crash.