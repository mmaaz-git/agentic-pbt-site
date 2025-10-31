# Bug Report: django.core.serializers.base.deserialize_m2m_values UnboundLocalError on Iterator Exception

**Target**: `django.core.serializers.base.deserialize_m2m_values`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `deserialize_m2m_values` function crashes with an `UnboundLocalError` when an iterator raises an exception before assigning any value to the loop variable `pk`, preventing proper error handling and masking the original exception.

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


if __name__ == "__main__":
    test_deserialize_m2m_values_handles_iteration_errors()
```

<details>

<summary>
**Failing input**: `error_type=RuntimeError`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 51, in <module>
    test_deserialize_m2m_values_handles_iteration_errors()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 18, in test_deserialize_m2m_values_handles_iteration_errors
    @settings(max_examples=20)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 47, in test_deserialize_m2m_values_handles_iteration_errors
    pytest.fail(f"UnboundLocalError raised. This is a bug in deserialize_m2m_values.")
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: UnboundLocalError raised. This is a bug in deserialize_m2m_values.
Falsifying example: test_deserialize_m2m_values_handles_iteration_errors(
    error_type=RuntimeError,
)
```
</details>

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

try:
    result = deserialize_m2m_values(field, field_value, using='default', handle_forward_references=False)
    print("No error occurred (unexpected)")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
UnboundLocalError: cannot access local variable 'pk' where it is not associated with a value
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/serializers/base.py", line 353, in deserialize_m2m_values
    for pk in pks_iter:
              ^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/repo.py", line 9, in __next__
    raise RuntimeError("Error during iteration")
RuntimeError: Error during iteration

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/repo.py", line 42, in <module>
    result = deserialize_m2m_values(field, field_value, using='default', handle_forward_references=False)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/serializers/base.py", line 360, in deserialize_m2m_values
    raise M2MDeserializationError(e, pk)
                                     ^^
UnboundLocalError: cannot access local variable 'pk' where it is not associated with a value
Exception type: UnboundLocalError
Exception message: cannot access local variable 'pk' where it is not associated with a value
```
</details>

## Why This Is A Bug

This violates Django's error handling contract for deserialization. The `M2MDeserializationError` class is specifically designed to wrap exceptions with context about the problematic value, as documented in its constructor which expects both `original_exc` and `pk` parameters. When the iterator raises an exception on its first `__next__()` call (line 353), the loop variable `pk` is never assigned a value. The exception handler at line 360 then attempts to reference this undefined `pk` variable, causing an `UnboundLocalError` instead of the intended `M2MDeserializationError`.

This breaks the documented error propagation chain where M2MDeserializationError should be caught by the caller and re-wrapped as `DeserializationError.WithData` (as seen in base.py:284-286). Instead, users receive a confusing UnboundLocalError that completely masks the original exception, making it impossible to diagnose the actual problem in their data or iterators.

## Relevant Context

The `deserialize_m2m_values` function is part of Django's core serialization framework, used extensively by the `loaddata` management command and fixture loading during tests. The function processes many-to-many field values during deserialization, supporting both natural keys and primary keys.

The bug affects real-world scenarios including:
- Loading fixtures with corrupted M2M data
- Database connection issues during deserialization
- Custom iterators that validate data before yielding values
- Mock objects used in testing that simulate error conditions

The M2MDeserializationError class definition at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/serializers/base.py:40-45` shows it expects both parameters to be available for proper error reporting.

## Proposed Fix

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