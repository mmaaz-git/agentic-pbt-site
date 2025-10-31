# Bug Report: django.dispatch.Signal WeakKeyDictionary Incompatibility

**Target**: `django.dispatch.Signal`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `Signal(use_caching=True)` is used, the signal crashes with `TypeError: cannot create weak reference to 'X' object` for many common sender types, including `None`, strings, and plain `object()` instances. This violates the documented API which explicitly allows `sender=None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.dispatch import Signal


def simple_receiver(**kwargs):
    return "received"


@given(st.booleans())
def test_connect_disconnect_inverse(use_caching):
    signal = Signal(use_caching=use_caching)
    signal.connect(simple_receiver)
    assert signal.has_listeners()
    signal.disconnect(simple_receiver)
    assert not signal.has_listeners()
```

**Failing input**: `use_caching=True` with `sender=None`

## Reproducing the Bug

```python
from django.dispatch import Signal


def simple_receiver(**kwargs):
    return "received"


signal = Signal(use_caching=True)
signal.connect(simple_receiver)
signal.send(sender=None)
```

**Error:**
```
TypeError: cannot create weak reference to 'NoneType' object
```

**Additional failing cases:**
```python
signal.send(sender="test_string")
signal.send(sender=123)
signal.send(sender=object())
```

All produce similar `TypeError: cannot create weak reference to 'X' object` errors.

## Why This Is A Bug

1. **Violates documented API**: The `connect()` docstring explicitly states: "The sender to which the receiver should respond. Must either be a Python object, or None to receive events from any sender."

2. **Common use case broken**: `sender=None` is a fundamental feature allowing receivers to listen to all senders.

3. **Breaks with common types**: Strings, integers, and plain objects are reasonable sender types but all fail.

4. **Caching should be transparent**: The `use_caching` parameter should be a performance optimization that doesn't change the API surface.

## Root Cause

In `dispatcher.py:47`:
```python
self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
```

`WeakKeyDictionary` requires keys to be weak-referenceable (i.e., have `__weakref__` slot). Many common types don't support weak references:
- `None`
- `str`
- `int`
- `float`
- `tuple`
- `object()` instances (unless the class defines `__slots__` with `__weakref__`)

## Fix

Replace `WeakKeyDictionary` with a regular `dict` that uses weak references only for objects that support them, or use a different caching strategy.

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -44,7 +44,7 @@ class Signal:
         # 'sender_receivers_cache'. The cache is cleaned when .connect() or
         # .disconnect() is called and populated on send().
-        self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
+        self.sender_receivers_cache = {} if use_caching else {}
         self._dead_receivers = False

     def connect(self, receiver, sender=None, weak=True, dispatch_uid=None):
```

**Note**: This simple fix uses a regular dict instead of WeakKeyDictionary. The original intent of using weak references for cache keys may have been to avoid memory leaks, but the cache is already cleared on `connect()` and `disconnect()` calls (lines 117, 152), so sender objects won't accumulate indefinitely. A more sophisticated fix could use weak references only for objects that support them while falling back to strong references for others.