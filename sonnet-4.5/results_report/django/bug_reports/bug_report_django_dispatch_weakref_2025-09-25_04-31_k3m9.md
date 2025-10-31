# Bug Report: django.dispatch Signal WeakKeyDictionary Caching Issue

**Target**: `django.dispatch.Signal`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `Signal(use_caching=True)` is used, the signal crashes with `TypeError` when the sender is `None` or other non-weakref-able types (like `object()`), despite the API explicitly documenting that `None` is a valid sender.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from django.dispatch import Signal

@settings(max_examples=500)
@given(st.booleans())
def test_has_listeners_consistency(use_caching):
    signal = Signal(use_caching=use_caching)

    assert not signal.has_listeners()

    def receiver(**kwargs):
        pass

    signal.connect(receiver, weak=False)
    assert signal.has_listeners()

    signal.disconnect(receiver)
    assert not signal.has_listeners()
```

**Failing input**: `use_caching=True`

## Reproducing the Bug

```python
from django.dispatch import Signal

signal = Signal(use_caching=True)

def receiver(**kwargs):
    return "response"

signal.connect(receiver, weak=False)

signal.has_listeners()
```

Running this code produces:
```
TypeError: cannot create weak reference to 'NoneType' object
```

The crash also occurs with `signal.send(sender=object())`:
```python
from django.dispatch import Signal

signal = Signal(use_caching=True)

def receiver(**kwargs):
    return "response"

signal.connect(receiver, weak=False)

signal.send(sender=object())
```

This produces:
```
TypeError: cannot create weak reference to 'object' object
```

## Why This Is A Bug

1. **API Contract Violation**: The `has_listeners(sender=None)` method explicitly has `sender=None` as the default parameter, but calling it with `use_caching=True` crashes.

2. **Documentation Mismatch**: The `send()` method documentation states: "The sender of the signal. Either a specific object or None." - but using `None` crashes when caching is enabled.

3. **Production Usage**: Django's own model signals use `use_caching=True` (see `django/db/models/signals.py` lines 42-51), making this a real-world issue.

4. **Root Cause**: In `dispatcher.py` line 47, when `use_caching=True`, a `WeakKeyDictionary` is used for caching:
   ```python
   self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
   ```

   However, `WeakKeyDictionary` cannot hold weak references to certain types like `None`, `object()`, integers, strings, etc. When these are used as senders, the code crashes at line 425:
   ```python
   receivers = self.sender_receivers_cache.get(sender)
   ```

## Fix

The fix is to use a regular dict instead of WeakKeyDictionary when the sender is not weakref-able, or to avoid trying to use non-weakref-able senders as keys in the WeakKeyDictionary. Here's one approach:

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -421,6 +421,15 @@ class Signal:
         This checks for weak references and resolves them, then returning only
         live receivers.
         """
+        # Helper to safely get from cache, handling non-weakref-able senders
+        def safe_cache_get(cache, key):
+            if not cache:
+                return None
+            try:
+                return cache.get(key)
+            except TypeError:
+                # sender is not weakref-able, can't use cache
+                return None
+
         receivers = None
         if self.use_caching and not self._dead_receivers:
-            receivers = self.sender_receivers_cache.get(sender)
+            receivers = safe_cache_get(self.sender_receivers_cache, sender)
             # We could end up here with NO_RECEIVERS even if we do check this case in
@@ -437,7 +446,11 @@ class Signal:
                         receivers.append((receiver, is_async))
                 if self.use_caching:
                     if not receivers:
-                        self.sender_receivers_cache[sender] = NO_RECEIVERS
+                        try:
+                            self.sender_receivers_cache[sender] = NO_RECEIVERS
+                        except TypeError:
+                            # sender is not weakref-able, skip caching
+                            pass
                     else:
                         # Note, we must cache the weakref versions.
-                        self.sender_receivers_cache[sender] = receivers
+                        try:
+                            self.sender_receivers_cache[sender] = receivers
+                        except TypeError:
+                            # sender is not weakref-able, skip caching
+                            pass
@@ -182,7 +191,7 @@ class Signal:
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or safe_cache_get(self.sender_receivers_cache, sender) is NO_RECEIVERS
         ):
             return []
```

Note: The `safe_cache_get` helper needs to be accessible from both `send()` and `_live_receivers()` methods, so it should either be a method of the Signal class or defined appropriately.