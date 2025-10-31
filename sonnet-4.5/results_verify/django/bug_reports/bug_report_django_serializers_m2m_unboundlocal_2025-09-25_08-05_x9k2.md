# Bug Report: django.core.serializers.base.deserialize_m2m_values UnboundLocalError

**Target**: `django.core.serializers.base.deserialize_m2m_values`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `deserialize_m2m_values` function at `/django/core/serializers/base.py:328-361` raises an `UnboundLocalError` when an exception occurs during iteration before the loop variable `pk` is assigned. The bug occurs at line 360 where the exception handler references `pk`, but this variable may not exist if the exception happens before or during the first iteration of the loop.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.core.serializers.base import deserialize_m2m_values, M2MDeserializationError
import pytest


class ErrorIterator:
    def __init__(self, error_type):
        self.error_type = error_type

    def __iter__(self):
        return self

    def __next__(self):
        raise self.error_type("Error during iteration")


@given(st.sampled_from([RuntimeError, ValueError, TypeError, KeyError]))
@settings(max_examples=20)
def test_deserialize_m2m_values_handles_iteration_errors(error_type):
    class MockPK:
        def to_python(self, v):
            return int(v)

    class MockMeta:
        pk = MockPK()

    class MockDefaultManager:
        pass

    class MockModel:
        _meta = MockMeta()
        _default_manager = MockDefaultManager()

    class MockRemoteField:
        model = MockModel

    class MockField:
        remote_field = MockRemoteField()

    field = MockField()
    field_value = ErrorIterator(error_type)

    with pytest.raises((M2MDeserializationError, UnboundLocalError)) as exc_info:
        deserialize_m2m_values(field, field_value, using='default', handle_forward_references=False)

    if isinstance(exc_info.value, UnboundLocalError):
        pytest.fail(f"UnboundLocalError raised. This is a bug in deserialize_m2m_values.")
```

**Failing input**: Any exception type (RuntimeError, ValueError, TypeError, KeyError, etc.)

## Reproducing the Bug

```python
from django.core.serializers.base import deserialize_m2m_values


class ErrorIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise RuntimeError("Error during iteration")


class MockPK:
    def to_python(self, v):
        return int(v)


class MockMeta:
    pk = MockPK()


class MockDefaultManager:
    pass


class MockModel:
    _meta = MockMeta()
    _default_manager = MockDefaultManager()


class MockRemoteField:
    model = MockModel


class MockField:
    remote_field = MockRemoteField()


field = MockField()
field_value = ErrorIterator()

deserialize_m2m_values(field, field_value, using='default', handle_forward_references=False)
```

**Output:**
```
UnboundLocalError: cannot access local variable 'pk' where it is not associated with a value
```

## Why This Is A Bug

The `deserialize_m2m_values` function is designed to deserialize many-to-many field values and handle errors gracefully by wrapping them in `M2MDeserializationError`. However, the exception handler at line 360 references the loop variable `pk`:

```python
try:
    values = []
    for pk in pks_iter:
        values.append(m2m_convert(pk))
    return values
except Exception as e:
    if isinstance(e, ObjectDoesNotExist) and handle_forward_references:
        return DEFER_FIELD
    else:
        raise M2MDeserializationError(e, pk)  # Line 360: 'pk' may not be defined!
```

If an exception is raised:
1. Before the loop starts (impossible in this specific code path)
2. During the first call to `__next__()` before `pk` is assigned
3. Any time the iterator itself raises an exception

Then `pk` will not be defined, causing an `UnboundLocalError` instead of the intended `M2MDeserializationError`.

This violates the function's error handling contract:
- It should wrap exceptions in `M2MDeserializationError` with context about the failing value
- Instead, it raises a confusing `UnboundLocalError` that masks the original exception
- This makes debugging much harder for users

## Fix

```diff
--- a/django/core/serializers/base.py
+++ b/django/core/serializers/base.py
@@ -350,6 +350,7 @@ def deserialize_m2m_values(field, field_value, using, handle_forward_references
         raise M2MDeserializationError(e, field_value)
     try:
         values = []
+        pk = None
         for pk in pks_iter:
             values.append(m2m_convert(pk))
         return values
@@ -357,7 +358,7 @@ def deserialize_m2m_values(field, field_value, using, handle_forward_references
         if isinstance(e, ObjectDoesNotExist) and handle_forward_references:
             return DEFER_FIELD
         else:
-            raise M2MDeserializationError(e, pk)
+            raise M2MDeserializationError(e, pk if pk is not None else field_value)
```

The fix:
1. Initializes `pk = None` before the loop to ensure it's always defined
2. Passes `field_value` to `M2MDeserializationError` if `pk` was never assigned
3. Ensures the exception always contains meaningful context about the failing value