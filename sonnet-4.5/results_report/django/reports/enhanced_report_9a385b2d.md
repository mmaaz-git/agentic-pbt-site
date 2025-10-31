# Bug Report: django.dispatch.Signal Crashes with Cached Non-Weakly-Referenceable Senders

**Target**: `django.dispatch.Signal.send` with `use_caching=True`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `Signal` is initialized with `use_caching=True`, calling `send()` with non-weakly-referenceable senders (strings, integers, None, tuples, booleans) raises a `TypeError` due to the internal use of `WeakKeyDictionary` which cannot handle these types as keys.

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

<details>

<summary>
**Failing input**: `sender='0'`
</summary>
```
Traceback (most recent call last):
  File "<string>", line 29, in <module>
    test_signal_with_caching_and_string_sender()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "<string>", line 18, in test_signal_with_caching_and_string_sender
    def test_signal_with_caching_and_string_sender(sender):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "<string>", line 22, in test_signal_with_caching_and_string_sender
    responses = signal.send(sender=sender)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/dispatch/dispatcher.py", line 183, in send
    or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/weakref.py", line 452, in get
    return self.data.get(ref(key),default)
                         ~~~^^^^^
TypeError: cannot create weak reference to 'str' object
Falsifying example: test_signal_with_caching_and_string_sender(
    sender='0',
)
Test failed with error:
```
</details>

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


# Test with use_caching=True and string sender
signal = Signal(use_caching=True)
sender = "my_sender"

signal.connect(receiver, sender=sender, weak=False)

# This should raise TypeError
try:
    responses = signal.send(sender=sender)
    print(f"Success: responses = {responses}")
except TypeError as e:
    print(f"TypeError raised: {e}")
```

<details>

<summary>
TypeError raised when sending signal with string sender
</summary>
```
TypeError raised: cannot create weak reference to 'str' object
```
</details>

## Why This Is A Bug

The `Signal` class's caching mechanism has an inherent design flaw that prevents it from working with Python's built-in immutable types. When `use_caching=True` is passed to `Signal.__init__`, it creates a `WeakKeyDictionary` at line 47 of django/dispatch/dispatcher.py:

```python
self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
```

`WeakKeyDictionary` requires all keys to support weak references. However, many Python built-in types cannot be weakly referenced:
- Strings (`str`)
- Integers (`int`)
- Floats (`float`)
- Booleans (`bool`)
- None (`NoneType`)
- Tuples (`tuple`)

The crash occurs in the `send()` method at line 183 when it attempts to use the sender as a key in the `WeakKeyDictionary`:

```python
or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
```

This creates several problems:

1. **Delayed Error Manifestation**: The error doesn't occur during `connect()` when the receiver is registered, but only during `send()` when the signal is dispatched, making it harder to debug.

2. **Inconsistent Behavior**: The same code works perfectly with `use_caching=False` (the default) but crashes with `use_caching=True`, creating a confusing inconsistency.

3. **Common Use Cases Affected**: String senders are extremely common in Django applications (e.g., using model names or signal names as senders), making this a practical limitation.

4. **Undocumented Limitation**: The `use_caching` parameter itself is undocumented in Django's public API, and there's no warning about this limitation with certain sender types.

## Relevant Context

The `use_caching` parameter appears to be an internal optimization feature that is not documented in Django's public API documentation. It's only discoverable through source code inspection or IDE autocomplete. The parameter was likely designed for use with class instances or other objects that can be weakly referenced, not considering the common pattern of using strings or other immutable types as senders.

Testing shows that:
- With `use_caching=True` and a class instance sender: Works correctly
- With `use_caching=False` and any sender type: Works correctly
- With `use_caching=True` and built-in immutable types: Crashes with TypeError
- Interestingly, `frozenset` objects can be weakly referenced and work with caching

Django source: https://github.com/django/django/blob/main/django/dispatch/dispatcher.py

## Proposed Fix

Replace the `WeakKeyDictionary` with a regular dictionary to support all sender types:

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -44,7 +44,7 @@ class Signal:
         # distinct sender we cache the receivers that sender has in
         # 'sender_receivers_cache'. The cache is cleaned when .connect() or
         # .disconnect() is called and populated on send().
-        self.sender_receivers_cache = weakref.WeakKeyDictionary() if use_caching else {}
+        self.sender_receivers_cache = {} if use_caching else {}
         self._dead_receivers = False
```

This fix is acceptable because:
1. The cache is cleared on every `connect()` and `disconnect()` call (lines 117 and 152)
2. Senders are typically long-lived objects (models, classes, string constants)
3. The memory impact is minimal given the cache's frequent clearing
4. It enables consistent behavior regardless of sender type