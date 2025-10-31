# Bug Report: django.dispatch Signal Crashes with None and String Senders When Caching Enabled

**Target**: `django.dispatch.Signal`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Signal crashes when use_caching=True and sender is None or a string, due to WeakKeyDictionary being unable to create weak references to these types. This extends the existing bug with object() instances to cover explicitly documented and commonly used sender types.

## Property-Based Test

```python
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
```

**Failing input**: `use_caching=True` (sender=None is explicitly allowed per documentation)

## Reproducing the Bug

```python
from django.dispatch import Signal

signal = Signal(use_caching=True)

def my_receiver(sender, **kwargs):
    return "response"

signal.connect(my_receiver)

signal.send(sender=None)
```

Expected: Signal is sent successfully
Actual: `TypeError: cannot create weak reference to 'NoneType' object`

## Why This Is A Bug

The Signal.send() documentation at line 174 explicitly states that sender can be "Either a specific object or None". Additionally, strings are commonly used as senders throughout Django (e.g., model names, app labels).

However, when use_caching=True, the code uses WeakKeyDictionary at line 47:
```python
self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
```

WeakKeyDictionary cannot hold weak references to:
- None
- Strings (and other immutable built-in types like int, float, tuple)
- Many other common Python objects

When send() is called with these senders (lines 183, 232, 296), it crashes:
```python
or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
```

This makes the caching feature incompatible with documented and commonly-used sender types.

## Fix

The fix should handle TypeError when accessing or setting cache entries for non-weakrefable senders. The existing bug report for object() instances includes a comprehensive patch that would also fix these cases:

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -180,7 +180,11 @@
         Return a list of tuple pairs [(receiver, response), ... ].
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (self.use_caching and self._cache_get(sender) is NO_RECEIVERS)
         ):
             return []
+
+    def _cache_get(self, sender):
+        try:
+            return self.sender_receivers_cache.get(sender)
+        except TypeError:
+            return None
+
+    def _cache_set(self, sender, value):
+        try:
+            self.sender_receivers_cache[sender] = value
+        except TypeError:
+            pass
```

Then use `_cache_get()` and `_cache_set()` throughout instead of direct dictionary access.