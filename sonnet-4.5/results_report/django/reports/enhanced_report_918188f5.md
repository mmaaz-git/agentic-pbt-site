# Bug Report: django.dispatch.Signal WeakKeyDictionary TypeError with Non-Weakref-able Senders

**Target**: `django.dispatch.Signal`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Django's `Signal` class crashes with `TypeError` when `use_caching=True` and the sender is `None` or any non-weakref-able type, despite the API explicitly documenting `None` as a valid sender and using it as the default parameter.

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

# Run the test
if __name__ == "__main__":
    test_has_listeners_consistency()
```

<details>

<summary>
**Failing input**: `use_caching=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 22, in <module>
    test_has_listeners_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 5, in test_has_listeners_consistency
    @given(st.booleans())
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 9, in test_has_listeners_consistency
    assert not signal.has_listeners()
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
Falsifying example: test_has_listeners_consistency(
    use_caching=True,
)
```
</details>

## Reproducing the Bug

```python
from django.dispatch import Signal

# Case 1: has_listeners() with default sender=None
print("Test Case 1: has_listeners() with default sender=None")
print("-" * 50)

signal = Signal(use_caching=True)

def receiver(**kwargs):
    return "response"

signal.connect(receiver, weak=False)

try:
    result = signal.has_listeners()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "=" * 50 + "\n")

# Case 2: send() with sender=None
print("Test Case 2: send() with sender=None")
print("-" * 50)

signal2 = Signal(use_caching=True)

def receiver2(**kwargs):
    return "response"

signal2.connect(receiver2, weak=False)

try:
    result = signal2.send(sender=None)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "=" * 50 + "\n")

# Case 3: send() with sender=object()
print("Test Case 3: send() with sender=object()")
print("-" * 50)

signal3 = Signal(use_caching=True)

def receiver3(**kwargs):
    return "response"

signal3.connect(receiver3, weak=False)

try:
    result = signal3.send(sender=object())
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "=" * 50 + "\n")

# Case 4: Control - same operations with use_caching=False work fine
print("Test Case 4: Control - same operations with use_caching=False")
print("-" * 50)

signal4 = Signal(use_caching=False)

def receiver4(**kwargs):
    return "response"

signal4.connect(receiver4, weak=False)

try:
    result1 = signal4.has_listeners()
    print(f"has_listeners(): {result1}")

    result2 = signal4.send(sender=None)
    print(f"send(sender=None): {result2}")

    result3 = signal4.send(sender=object())
    print(f"send(sender=object()): {result3}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError crashes with non-weakref-able senders when caching is enabled
</summary>
```
Test Case 1: has_listeners() with default sender=None
--------------------------------------------------
Error: TypeError: cannot create weak reference to 'NoneType' object

==================================================

Test Case 2: send() with sender=None
--------------------------------------------------
Error: TypeError: cannot create weak reference to 'NoneType' object

==================================================

Test Case 3: send() with sender=object()
--------------------------------------------------
Error: TypeError: cannot create weak reference to 'object' object

==================================================

Test Case 4: Control - same operations with use_caching=False
--------------------------------------------------
has_listeners(): True
send(sender=None): [(<function receiver4 at 0x7b7b60e363e0>, 'response')]
send(sender=object()): [(<function receiver4 at 0x7b7b60e363e0>, 'response')]
```
</details>

## Why This Is A Bug

1. **API Contract Violation**: The `Signal.send()` method docstring at line 174 in `dispatcher.py` explicitly states: "The sender of the signal. Either a specific object or None." However, using `None` as sender crashes when `use_caching=True`.

2. **Default Parameter Failure**: The `has_listeners()` method at line 155 has `sender=None` as its default parameter. Calling this method without arguments crashes when caching is enabled, making the default parameter unusable.

3. **Inconsistent Behavior**: The same code works perfectly with `use_caching=False` but crashes with `use_caching=True`. This inconsistency is not documented anywhere.

4. **Production Impact**: Django's own model signals (`pre_init`, `post_init`, `pre_save`, `post_save`, `pre_delete`, `post_delete`, `m2m_changed`) all use `use_caching=True` (see `django/db/models/signals.py` lines 42-51), making this a real issue in production Django applications.

5. **Root Cause**: At line 47 in `dispatcher.py`, when `use_caching=True`, the code creates a `weakref.WeakKeyDictionary()`:
   ```python
   self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
   ```

   Python's `WeakKeyDictionary` cannot create weak references to certain types including:
   - `None` (NoneType)
   - Basic `object()` instances without `__weakref__` slot
   - Built-in immutable types (int, str, tuple, etc.)

   The crash occurs at line 425 when trying to access the cache:
   ```python
   receivers = self.sender_receivers_cache.get(sender)
   ```

   And at lines 440 and 443 when trying to set cache values:
   ```python
   self.sender_receivers_cache[sender] = NO_RECEIVERS
   self.sender_receivers_cache[sender] = receivers
   ```

## Relevant Context

This bug affects any Django application that:
- Uses signals with `use_caching=True` (including Django's built-in model signals)
- Calls signal methods with `sender=None` or non-weakref-able objects
- Relies on the documented behavior that `None` is a valid sender

The bug is particularly insidious because:
- It only manifests when caching is enabled, which is the case for Django's model signals
- The error message doesn't clearly indicate the problem is with the caching mechanism
- There's no documentation warning about this limitation

Relevant Django source files:
- `/django/dispatch/dispatcher.py` - Contains the Signal class with the bug
- `/django/db/models/signals.py` - Shows Django's own signals use `use_caching=True`

## Proposed Fix

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -1,6 +1,7 @@
 import asyncio
 import logging
 import threading
 import weakref
+from typing import Any, Optional

 from asgiref.sync import async_to_sync, iscoroutinefunction, sync_to_async

@@ -414,6 +415,20 @@ class Signal:
                 if not (isinstance(r[1], weakref.ReferenceType) and r[1]() is None)
             ]

+    def _safe_cache_get(self, cache, key, default=None):
+        """Safely get from cache, handling non-weakref-able keys."""
+        if not cache:
+            return default
+        try:
+            return cache.get(key, default)
+        except TypeError:
+            # Key is not weakref-able, can't use cache
+            return default
+
+    def _safe_cache_set(self, cache, key, value):
+        """Safely set in cache, handling non-weakref-able keys."""
+        try:
+            cache[key] = value
+        except TypeError:
+            # Key is not weakref-able, skip caching
+            pass
+
     def _live_receivers(self, sender):
         """
         Filter sequence of receivers to get resolved, live receivers.
@@ -422,7 +437,7 @@ class Signal:
         """
         receivers = None
         if self.use_caching and not self._dead_receivers:
-            receivers = self.sender_receivers_cache.get(sender)
+            receivers = self._safe_cache_get(self.sender_receivers_cache, sender)
             # We could end up here with NO_RECEIVERS even if we do check this case in
             # .send() prior to calling _live_receivers() due to concurrent .send() call.
             if receivers is NO_RECEIVERS:
@@ -437,10 +452,10 @@ class Signal:
                         receivers.append((receiver, is_async))
                 if self.use_caching:
                     if not receivers:
-                        self.sender_receivers_cache[sender] = NO_RECEIVERS
+                        self._safe_cache_set(self.sender_receivers_cache, sender, NO_RECEIVERS)
                     else:
                         # Note, we must cache the weakref versions.
-                        self.sender_receivers_cache[sender] = receivers
+                        self._safe_cache_set(self.sender_receivers_cache, sender, receivers)
         non_weak_sync_receivers = []
         non_weak_async_receivers = []
         for receiver, is_async in receivers:
@@ -181,7 +196,7 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or self._safe_cache_get(self.sender_receivers_cache, sender) is NO_RECEIVERS
         ):
             return []
         responses = []
@@ -231,7 +246,7 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or self._safe_cache_get(self.sender_receivers_cache, sender) is NO_RECEIVERS
         ):
             return []
         sync_receivers, async_receivers = self._live_receivers(sender)
@@ -294,7 +309,7 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or self._safe_cache_get(self.sender_receivers_cache, sender) is NO_RECEIVERS
         ):
             return []

@@ -359,7 +374,7 @@ class Signal:
         """
         if (
             not self.receivers
-            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
+            or self._safe_cache_get(self.sender_receivers_cache, sender) is NO_RECEIVERS
         ):
             return []
```