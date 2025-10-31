# Bug Report: django.dispatch.Signal WeakKeyDictionary Breaks Common Sender Types

**Target**: `django.dispatch.Signal`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `Signal(use_caching=True)` is enabled, the signal crashes with `TypeError: cannot create weak reference` for many common sender types including `None`, strings, integers, and plain objects, violating the documented API contract.

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


if __name__ == "__main__":
    test_connect_disconnect_inverse()
```

<details>

<summary>
**Failing input**: `use_caching=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 19, in <module>
    test_connect_disconnect_inverse()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 10, in test_connect_disconnect_inverse
    def test_connect_disconnect_inverse(use_caching):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 13, in test_connect_disconnect_inverse
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
Falsifying example: test_connect_disconnect_inverse(
    use_caching=True,
)
```
</details>

## Reproducing the Bug

```python
from django.dispatch import Signal


def simple_receiver(**kwargs):
    return "received"


# Test case 1: sender=None with use_caching=True
print("Test 1: sender=None with use_caching=True")
try:
    signal = Signal(use_caching=True)
    signal.connect(simple_receiver)
    signal.send(sender=None)
    print("SUCCESS: No error occurred")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 2: sender="test_string" with use_caching=True
print("\nTest 2: sender='test_string' with use_caching=True")
try:
    signal = Signal(use_caching=True)
    signal.connect(simple_receiver)
    signal.send(sender="test_string")
    print("SUCCESS: No error occurred")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 3: sender=123 with use_caching=True
print("\nTest 3: sender=123 with use_caching=True")
try:
    signal = Signal(use_caching=True)
    signal.connect(simple_receiver)
    signal.send(sender=123)
    print("SUCCESS: No error occurred")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 4: sender=object() with use_caching=True
print("\nTest 4: sender=object() with use_caching=True")
try:
    signal = Signal(use_caching=True)
    signal.connect(simple_receiver)
    signal.send(sender=object())
    print("SUCCESS: No error occurred")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 5: All cases work with use_caching=False
print("\nTest 5: All sender types with use_caching=False")
try:
    signal = Signal(use_caching=False)
    signal.connect(simple_receiver)
    signal.send(sender=None)
    signal.send(sender="test_string")
    signal.send(sender=123)
    signal.send(sender=object())
    print("SUCCESS: All sender types work without caching")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError crashes for all common sender types when caching is enabled
</summary>
```
Test 1: sender=None with use_caching=True
ERROR: TypeError: cannot create weak reference to 'NoneType' object

Test 2: sender='test_string' with use_caching=True
ERROR: TypeError: cannot create weak reference to 'str' object

Test 3: sender=123 with use_caching=True
ERROR: TypeError: cannot create weak reference to 'int' object

Test 4: sender=object() with use_caching=True
ERROR: TypeError: cannot create weak reference to 'object' object

Test 5: All sender types with use_caching=False
SUCCESS: All sender types work without caching
```
</details>

## Why This Is A Bug

This bug violates Django's documented API contract in multiple critical ways:

1. **Explicit Documentation Violation**: The `connect()` method's docstring at line 69-71 of `dispatcher.py` explicitly states:
   > "sender: The sender to which the receiver should respond. Must either be a Python object, or None to receive events from any sender."

   The phrase "Python object" is inclusive and doesn't restrict to weak-referenceable objects. More importantly, `sender=None` is specifically documented as a valid and important use case for receiving events from any sender.

2. **Breaking Core Functionality**: The ability to use `sender=None` is fundamental to Django's signal system. It's the standard way to create receivers that listen to all senders, which is a common pattern in Django applications.

3. **Unexpected API Change**: The `use_caching` parameter is presented as a performance optimization without any documented restrictions. Users reasonably expect it to be a transparent optimization that doesn't change which inputs are valid.

4. **Common Type Failures**: The bug affects not just `None`, but also strings, integers, and plain object instances - all reasonable sender types that developers might use in their applications.

5. **Silent Contract Change**: Nothing in the Signal constructor or its documentation warns that enabling caching restricts sender types. This creates a hidden pitfall where code that works fine without caching suddenly crashes when caching is enabled.

## Relevant Context

The root cause is in `dispatcher.py:47`:
```python
self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
```

`WeakKeyDictionary` requires all keys to support weak references. Python's built-in types like `None`, `str`, `int`, `float`, `tuple`, and plain `object()` instances (without `__weakref__` slot) cannot be weak referenced. This limitation of `WeakKeyDictionary` makes it incompatible with Django's documented signal API.

The error occurs when the cache is accessed in `_live_receivers()` method at line 425:
```python
receivers = self.sender_receivers_cache.get(sender)
```

Even the `has_listeners()` method fails, not just `send()`, making it impossible to even check if a signal has listeners when caching is enabled with these sender types.

Documentation references:
- Django dispatch source: `/django/dispatch/dispatcher.py`
- WeakKeyDictionary docs: https://docs.python.org/3/library/weakref.html#weakref.WeakKeyDictionary

## Proposed Fix

Replace `WeakKeyDictionary` with a hybrid approach that handles both weak-referenceable and non-weak-referenceable senders:

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -44,7 +44,23 @@ class Signal:
         # distinct sender we cache the receivers that sender has in
         # 'sender_receivers_cache'. The cache is cleaned when .connect() or
         # .disconnect() is called and populated on send().
-        self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
+        if use_caching:
+            # Use a hybrid cache that supports both weak-referenceable and
+            # non-weak-referenceable senders
+            class HybridWeakKeyDict(dict):
+                def __setitem__(self, key, value):
+                    try:
+                        # Try to create a weak reference
+                        key = weakref.ref(key)
+                    except TypeError:
+                        # Key doesn't support weak references, use it directly
+                        pass
+                    super().__setitem__(key, value)
+
+                def get(self, key, default=None):
+                    # Try both weak and strong reference
+                    try:
+                        return super().get(weakref.ref(key), super().get(key, default))
+                    except TypeError:
+                        return super().get(key, default)
+
+            self.sender_receivers_cache = HybridWeakKeyDict()
+        else:
+            self.sender_receivers_cache = {}
         self._dead_receivers = False
```