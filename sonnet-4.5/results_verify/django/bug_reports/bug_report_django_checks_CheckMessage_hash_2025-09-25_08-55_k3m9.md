# Bug Report: CheckMessage Hash/Equality Contract Violation

**Target**: `django.core.checks.CheckMessage`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CheckMessage` class implements `__eq__()` but not `__hash__()`, violating Python's equality/hash contract and making instances unhashable. This prevents `CheckMessage` objects from being used in sets or as dictionary keys.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.checks import CheckMessage

@st.composite
def check_messages(draw):
    level = draw(st.integers(min_value=1, max_value=100))
    msg = draw(st.text())
    hint = draw(st.one_of(st.none(), st.text()))
    id_val = draw(st.one_of(st.none(), st.text(min_size=1)))
    return CheckMessage(level, msg, hint=hint, id=id_val)

@given(check_messages())
def test_equal_objects_have_same_hash(msg):
    msg_copy = CheckMessage(msg.level, msg.msg, hint=msg.hint, obj=msg.obj, id=msg.id)
    if msg == msg_copy:
        assert hash(msg) == hash(msg_copy)
```

**Failing input**: Any `CheckMessage` instance

## Reproducing the Bug

```python
from django.core.checks import CheckMessage, ERROR

msg1 = CheckMessage(ERROR, "Test error")
msg2 = CheckMessage(ERROR, "Test error")

print(msg1 == msg2)

hash(msg1)

s = {msg1, msg2}
```

**Output:**
```
True
Traceback (most recent call last):
  File "reproduce.py", line 8, in <module>
    hash(msg1)
TypeError: unhashable type: 'CheckMessage'
```

## Why This Is A Bug

According to Python's data model documentation: "If a class defines mutable objects and implements an `__eq__()` method, it should not implement `__hash__()`, since the implementation of hashable collections requires that a key's hash value is immutable." However, `CheckMessage` objects are effectively immutable (no methods modify state), so they should be hashable.

More importantly, Python's hash/equality contract states: "If two objects compare equal, they must have the same hash value." By implementing `__eq__()` without `__hash__()`, Python automatically sets `__hash__ = None`, making the objects unhashable. This prevents legitimate use cases like:
- Deduplicating check messages using sets: `unique_msgs = set(all_messages)`
- Using messages as dictionary keys for categorization
- Any container operation requiring hashability

## Fix

```diff
--- a/django/core/checks/messages.py
+++ b/django/core/checks/messages.py
@@ -22,6 +22,11 @@ class CheckMessage:
             for attr in ["level", "msg", "hint", "obj", "id"]
         )

+    def __hash__(self):
+        return hash((self.level, self.msg, self.hint, self.obj, self.id))
+
     def __str__(self):
         from django.db import models
```

Note: The `obj` attribute might not always be hashable (it could be a model instance). A more robust fix would handle unhashable `obj` values:

```diff
--- a/django/core/checks/messages.py
+++ b/django/core/checks/messages.py
@@ -22,6 +22,14 @@ class CheckMessage:
             for attr in ["level", "msg", "hint", "obj", "id"]
         )

+    def __hash__(self):
+        try:
+            obj_hash = hash(self.obj)
+        except TypeError:
+            obj_hash = id(self.obj)
+        return hash((self.level, self.msg, self.hint, obj_hash, self.id))
+
     def __str__(self):
         from django.db import models
```