# Bug Report: pydantic.experimental.pipeline == and != Operators

**Target**: `pydantic.experimental.pipeline._Pipeline`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `==` and `!=` operators on `_Pipeline` return `bool` instead of `_Pipeline` objects, inconsistent with the `.eq()` and `.not_eq()` methods which correctly return `_Pipeline` objects.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.experimental.pipeline import _Pipeline


@given(st.integers())
def test_eq_operator_returns_pipeline_not_bool(x):
    pipeline = _Pipeline(())

    method_result = pipeline.eq(x)
    operator_result = pipeline == x

    assert isinstance(method_result, _Pipeline), \
        f"pipeline.eq({x}) should return _Pipeline, got {type(method_result)}"

    assert isinstance(operator_result, _Pipeline), \
        f"pipeline == {x} should return _Pipeline, got {type(operator_result)}"


@given(st.integers())
def test_ne_operator_returns_pipeline_not_bool(x):
    pipeline = _Pipeline(())

    method_result = pipeline.not_eq(x)
    operator_result = pipeline != x

    assert isinstance(method_result, _Pipeline), \
        f"pipeline.not_eq({x}) should return _Pipeline, got {type(method_result)}"

    assert isinstance(operator_result, _Pipeline), \
        f"pipeline != {x} should return _Pipeline, got {type(operator_result)}"
```

**Failing input**: Any integer value (e.g., `x=0`)

## Reproducing the Bug

```python
from pydantic.experimental.pipeline import _Pipeline

pipeline = _Pipeline(())

print(f"pipeline.eq(5): {type(pipeline.eq(5))}")
print(f"pipeline == 5: {type(pipeline == 5)}")

print(f"pipeline.not_eq(5): {type(pipeline.not_eq(5))}")
print(f"pipeline != 5: {type(pipeline != 5)}")
```

Output:
```
pipeline.eq(5): <class 'pydantic.experimental.pipeline._Pipeline'>
pipeline == 5: <class 'bool'>
pipeline.not_eq(5): <class 'pydantic.experimental.pipeline._Pipeline'>
pipeline != 5: <class 'bool'>
```

## Why This Is A Bug

The API is inconsistent:
- `.eq()` method returns `_Pipeline` (correct)
- `==` operator returns `bool` (incorrect)
- `.not_eq()` method returns `_Pipeline` (correct)
- `!=` operator returns `bool` (incorrect)

This violates the principle of least surprise and makes the API confusing. Users naturally expect `pipeline == x` to work like `pipeline.eq(x)`, but instead it performs a regular equality comparison.

## Fix

Add `__eq__` and `__ne__` methods to the `_Pipeline` class:

```diff
--- a/pipeline.py
+++ b/pipeline.py
@@ -145,6 +145,12 @@ class _Pipeline(Generic[_InT, _OutT]):
     def eq(self, value: _OutT) -> _Pipeline[_InT, _OutT]:
         return self.constrain(_Eq(value))

+    def __eq__(self, value: object) -> _Pipeline[_InT, _OutT]:
+        return self.eq(value)
+
+    def __ne__(self, value: object) -> _Pipeline[_InT, _OutT]:
+        return self.not_eq(value)
+
     def not_eq(self, value: _OutT) -> _Pipeline[_InT, _OutT]:
         return self.constrain(_NotEq(value))
```