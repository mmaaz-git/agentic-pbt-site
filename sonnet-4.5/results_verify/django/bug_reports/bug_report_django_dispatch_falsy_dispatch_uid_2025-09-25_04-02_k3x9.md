# Bug Report: Django Signal dispatch_uid Falsy Value Handling

**Target**: `django.dispatch.Signal.connect` and `django.dispatch.Signal.disconnect`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Signal class incorrectly handles falsy but valid `dispatch_uid` values (such as 0, empty string, or False) due to using truthiness checks instead of explicit None checks. This causes receivers registered with falsy dispatch_uids to be incorrectly keyed and become undisconnectable using the dispatch_uid alone.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.dispatch import Signal

def dummy_receiver(**kwargs):
    return "received"

@given(st.integers(min_value=0, max_value=100))
def test_signal_disconnect_with_dispatch_uid(n):
    signal = Signal()

    for i in range(n):
        signal.connect(dummy_receiver, dispatch_uid=i)

    for i in range(n):
        result = signal.disconnect(receiver=None, sender=None, dispatch_uid=i)
        assert result, f"Should be able to disconnect receiver {i}"
```

**Failing input**: `n=1` (which uses `dispatch_uid=0`)

## Reproducing the Bug

```python
from django.dispatch import Signal

def dummy_receiver(**kwargs):
    return "received"

signal = Signal()

signal.connect(dummy_receiver, dispatch_uid=0)
print(f"Connected. Has listeners: {signal.has_listeners()}")

result = signal.disconnect(dispatch_uid=0)
print(f"Disconnect result: {result}")
print(f"Still has listeners: {signal.has_listeners()}")
```

**Expected**: Disconnect returns `True`, signal has no listeners
**Actual**: Disconnect returns `False`, signal still has the listener

## Why This Is A Bug

The issue is in `dispatcher.py` at lines 96-99 and 138-141:

```python
if dispatch_uid:
    lookup_key = (dispatch_uid, _make_id(sender))
else:
    lookup_key = (_make_id(receiver), _make_id(sender))
```

When `dispatch_uid=0` (or any falsy value), the condition `if dispatch_uid:` evaluates to `False`, causing the code to use the receiver-based lookup_key instead. However, when disconnecting with `dispatch_uid=0` and `receiver=None`, the lookup keys don't match:

- **Connect** with `dispatch_uid=0`: lookup_key = `(_make_id(dummy_receiver), _make_id(None))`
- **Disconnect** with `dispatch_uid=0, receiver=None`: lookup_key = `(_make_id(None), _make_id(None))`

These keys don't match, so disconnect fails.

The docstring for `disconnect` explicitly states: "receiver: The registered receiver to disconnect. May be none if dispatch_uid is specified." This implies that `dispatch_uid` alone should be sufficient for disconnection, but the implementation doesn't support this for falsy values.

## Fix

```diff
--- a/django/dispatch/dispatcher.py
+++ b/django/dispatch/dispatcher.py
@@ -93,7 +93,7 @@ class Signal:
                     "Signal receivers must accept keyword arguments (**kwargs)."
                 )

-        if dispatch_uid:
+        if dispatch_uid is not None:
             lookup_key = (dispatch_uid, _make_id(sender))
         else:
             lookup_key = (_make_id(receiver), _make_id(sender))
@@ -135,7 +135,7 @@ class Signal:
             dispatch_uid
                 the unique identifier of the receiver to disconnect
         """
-        if dispatch_uid:
+        if dispatch_uid is not None:
             lookup_key = (dispatch_uid, _make_id(sender))
         else:
             lookup_key = (_make_id(receiver), _make_id(sender))
```