# Bug Report: django.dispatch.Signal WeakKeyDictionary with Non-Weakref Senders

**Target**: `django.dispatch.Signal`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `Signal(use_caching=True)` is instantiated, the signal crashes with `TypeError: cannot create weak reference to 'object' object` when using common Python objects (like `object()` instances) as senders.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from unittest.mock import Mock
from django.dispatch import Signal

@given(st.booleans())
def test_connect_disconnect_inverse(use_caching):
    signal = Signal(use_caching=use_caching)
    recv = Mock(return_value=None, __name__="test_receiver")
    sender = object()

    signal.connect(recv, sender=sender, weak=False)
    assert signal.has_listeners(sender=sender)

    disconnected = signal.disconnect(recv, sender=sender)
    assert disconnected
    assert not signal.has_listeners(sender=sender)
```

**Failing input**: `use_caching=True`

## Reproducing the Bug

```python
from django.dispatch import Signal

signal = Signal(use_caching=True)
sender = object()

def receiver(**kwargs):
    return 42

signal.connect(receiver, sender=sender, weak=False)

signal.has_listeners(sender=sender)
```

## Why This Is A Bug

The `Signal` class documentation states that sender should accept "any Python object" (line 70-71 in dispatcher.py). However, when `use_caching=True`, the implementation uses `weakref.WeakKeyDictionary()` (line 47) to store the sender cache, which can only accept objects that support weak references.

Many common Python objects cannot be weakly referenced:
- `object()` instances
- integers, strings, tuples
- None

This violates the documented API contract and makes `use_caching=True` unusable with many valid senders.

## Fix

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -44,7 +44,7 @@ class Signal:
         # 'sender_receivers_cache'. The cache is cleaned when .connect() or
         # .disconnect() is called and populated on send().
-        self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
+        self.sender_receivers_cache = {} if not use_caching else {}
         self._dead_receivers = False
```

Note: The WeakKeyDictionary was likely intended to prevent memory leaks when sender objects are garbage collected. However, this creates a correctness bug. A better solution would be to use a regular dict and accept that senders will be kept alive while the signal exists, or implement a hybrid approach that uses weak references when possible and falls back to strong references otherwise.