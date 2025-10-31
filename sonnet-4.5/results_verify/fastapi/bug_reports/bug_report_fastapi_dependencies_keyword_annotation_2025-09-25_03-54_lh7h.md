# Bug Report: fastapi.dependencies.utils - Crash on Keyword String Annotations

**Target**: `fastapi.dependencies.utils.get_typed_annotation`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

FastAPI crashes with `SyntaxError` when processing endpoint functions that use Python keywords as string type annotations (e.g., `def foo(x: "if"): pass`). While unusual, such annotations are syntactically valid Python and should not cause FastAPI to crash.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from fastapi.dependencies.utils import get_typed_annotation
import keyword


@settings(max_examples=200)
@given(
    st.text(min_size=1, max_size=50).filter(lambda s: s.isidentifier()),
)
def test_get_typed_annotation_handles_keywords(type_str):
    assume(keyword.iskeyword(type_str))

    try:
        result = get_typed_annotation(type_str, {})
    except (NameError, AttributeError):
        pass
    except SyntaxError:
        assert False, f"SyntaxError should not be raised for valid identifier '{type_str}'"
```

**Failing input**: `'if'` (or any Python keyword: 'class', 'def', 'for', etc.)

## Reproducing the Bug

```python
from fastapi.dependencies.utils import get_dependant

def endpoint_function(x: "if"):
    return {"x": x}

dependant = get_dependant(path="/test", call=endpoint_function)
```

**Output:**
```
SyntaxError: Forward reference must be an expression -- got 'if'
```

## Why This Is A Bug

1. Python allows keywords as string annotations: `def foo(x: "if"): pass` is valid Python code
2. Users can write such code (even if accidentally or for testing purposes)
3. FastAPI should handle this gracefully, either by:
   - Treating it as `Any` type
   - Raising a helpful `ValueError` or `TypeError` with a clear message
   - Skipping type resolution for invalid forward references
4. Crashing with `SyntaxError` provides a poor user experience

The bug occurs in `get_typed_annotation()` at line 249:
```python
def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)  # Crashes here on keywords
        annotation = evaluate_forwardref(annotation, globalns, globalns)
    return annotation
```

`ForwardRef` validates that the string is a valid Python expression, which excludes keywords.

## Fix

```diff
--- a/fastapi/dependencies/utils.py
+++ b/fastapi/dependencies/utils.py
@@ -246,7 +246,12 @@ def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:

 def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
     if isinstance(annotation, str):
-        annotation = ForwardRef(annotation)
+        try:
+            annotation = ForwardRef(annotation)
+        except SyntaxError:
+            # Invalid forward reference (e.g., Python keyword), treat as Any
+            return Any
+
         annotation = evaluate_forwardref(annotation, globalns, globalns)
     return annotation
```

Alternatively, for stricter validation, raise a more informative error:

```diff
--- a/fastapi/dependencies/utils.py
+++ b/fastapi/dependencies/utils.py
@@ -246,7 +246,14 @@ def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:

 def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
     if isinstance(annotation, str):
-        annotation = ForwardRef(annotation)
+        try:
+            annotation = ForwardRef(annotation)
+        except SyntaxError as e:
+            raise TypeError(
+                f"Invalid type annotation string {annotation!r}: {e}. "
+                "Type annotations must be valid Python expressions."
+            ) from e
+
         annotation = evaluate_forwardref(annotation, globalns, globalns)
     return annotation
```