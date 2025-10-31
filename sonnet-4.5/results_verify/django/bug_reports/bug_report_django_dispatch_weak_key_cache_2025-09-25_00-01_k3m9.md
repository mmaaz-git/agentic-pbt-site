# Bug Report: django.dispatch.Signal Crashes with Caching and Non-Weakly-Referenceable Senders

**Target**: `django.dispatch.Signal.send` with `use_caching=True`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When a `Signal` is created with `use_caching=True`, calling `send()` with a non-weakly-referenceable sender (such as strings, integers, None, tuples) raises a `TypeError`. This affects common use cases where string senders are used.

## Property-Based Test

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=False)

from hypothesis import given, strategies as st
from django.dispatch import Signal


def receiver(**kwargs):
    return "received"


@given(st.text(min_size=1))
def test_signal_with_caching_and_string_sender(sender):
    signal = Signal(use_caching=True)

    signal.connect(receiver, sender=sender, weak=False)
    responses = signal.send(sender=sender)

    assert len(responses) > 0
```

**Failing input**: `sender='0'` (or any string value)

## Reproducing the Bug

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=False)

from django.dispatch import Signal


def receiver(**kwargs):
    return "received"


signal = Signal(use_caching=True)
sender = "my_sender"

signal.connect(receiver, sender=sender, weak=False)
responses = signal.send(sender=sender)
```

**Error**:
```
TypeError: cannot create weak reference to 'str' object
```

## Why This Is A Bug

The `Signal.__init__` method creates a `WeakKeyDictionary` when `use_caching=True`:

```python
self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
```

However, `WeakKeyDictionary` requires keys to be weakly referenceable. Many common Python types cannot be weakly referenced, including:
- `str`, `int`, `float`, `bool`, `None`
- `tuple`, `frozenset`
- Other built-in immutable types

When `send()` is called with such a sender, it tries to use the sender as a key in the `WeakKeyDictionary` at line 183:

```python
or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
```

This raises a `TypeError` for non-weakly-referenceable types.

This is particularly problematic because:
1. The error only occurs during `send()`, not `connect()`
2. String senders are common in Django (e.g., model names, signal names)
3. The documentation doesn't warn about this limitation
4. The same code works fine with `use_caching=False`

## Fix

Use a regular dictionary instead of `WeakKeyDictionary` for the cache. While this means cached senders won't be garbage collected, this is acceptable because:
1. The cache is cleared on every `connect()` and `disconnect()` call
2. Senders are typically long-lived objects anyway (models, classes, strings)
3. The fix enables caching to work with all sender types

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -44,7 +44,7 @@ class Signal:
         # distinct sender we cache the receivers that sender has in
         # 'sender_receivers_cache'. The cache is cleaned when .connect() or
         # .disconnect() is called and populated on send().
-        self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
+        self.sender_receivers_cache = {}
         self._dead_receivers = False
```

Alternatively, if WeakKeyDictionary behavior is desired for weakly-referenceable senders, the code could gracefully fall back:

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -180,7 +180,14 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or (
+                self.use_caching
+                and isinstance(self.sender_receivers_cache, weakref.WeakKeyDictionary)
+                and (
+                    try_weakref_get(self.sender_receivers_cache, sender)
+                    is NO_RECEIVERS
+                )
+            )
         ):
```

However, the simpler fix is to just use a regular dictionary, as the cache is cleared frequently anyway.