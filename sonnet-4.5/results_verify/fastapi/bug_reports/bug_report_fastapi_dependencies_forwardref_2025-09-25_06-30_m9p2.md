# Bug Report: fastapi.dependencies get_typed_annotation Unsafe Evaluation

**Target**: `fastapi.dependencies.utils.get_typed_annotation`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`get_typed_annotation()` evaluates string annotations as Python code without proper validation, which can lead to crashes (ZeroDivisionError, etc.) when invalid expressions are provided as type annotations.

## Property-Based Test

```python
from hypothesis import assume, given, strategies as st
from fastapi.dependencies.utils import get_typed_annotation


@given(st.text(min_size=1, max_size=50))
def test_get_typed_annotation_invalid_forward_ref(annotation_str):
    assume(not annotation_str.isidentifier())
    assume('.' not in annotation_str)

    result = get_typed_annotation(annotation_str, {})
    assert result is not None
```

**Failing input**: `"0/0"`

## Reproducing the Bug

```python
from fastapi.dependencies.utils import get_typed_annotation

result = get_typed_annotation("0/0", {})
```

Running this code results in:
```
ZeroDivisionError: division by zero
```

## Why This Is A Bug

The function converts string annotations to ForwardRef objects and then evaluates them using `evaluate_forwardref`. This evaluation uses Python's `eval()` internally, which executes arbitrary Python expressions. When the string contains invalid arithmetic like `"0/0"`, it crashes with ZeroDivisionError instead of gracefully handling the invalid annotation.

While in typical usage, annotations come from function definitions (developer-controlled), there are edge cases where invalid annotations could be passed, leading to unexpected crashes.

## Fix

The function should wrap the evaluation in a try-except block to handle evaluation errors gracefully:

```diff
def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
-       annotation = evaluate_forwardref(annotation, globalns, globalns)
+       try:
+           annotation = evaluate_forwardref(annotation, globalns, globalns)
+       except Exception:
+           # If evaluation fails, return the original ForwardRef
+           # or raise a more descriptive error
+           pass
    return annotation
```

Alternatively, validate that the annotation string is a valid identifier or type expression before evaluation.