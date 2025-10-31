# Bug Report: fastapi.dependencies.utils.get_typed_annotation Crashes with Keyword String Annotations

**Target**: `fastapi.dependencies.utils.get_typed_annotation`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_typed_annotation` function crashes with a `SyntaxError` when processing string annotations that are Python keywords (e.g., "if", "class", "def"). This causes FastAPI applications to fail during startup if a route handler has a parameter with such an annotation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.dependencies.utils import get_typed_annotation
import keyword

@given(annotation_str=st.sampled_from(keyword.kwlist))
def test_get_typed_annotation_handles_keywords(annotation_str):
    globalns = {}
    result = get_typed_annotation(annotation_str, globalns)
```

**Failing input**: `annotation_str='if'` (and most other Python keywords)

## Reproducing the Bug

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/test")
def test_endpoint(x: "if"):
    return {"x": x}
```

Running this code produces:

```
SyntaxError: Forward reference must be an expression -- got 'if'
```

## Why This Is A Bug

1. Python allows string annotations with any content, including keywords: `def f(x: "if"): pass` is syntactically valid
2. FastAPI processes these annotations during route registration via `get_dependant` → `get_typed_signature` → `get_typed_annotation`
3. The function unconditionally creates a `ForwardRef` from string annotations without validating that the string is a valid expression
4. This causes the application to crash during startup rather than providing a helpful error message

While using keywords as type annotations is unusual, it can happen accidentally (e.g., typos, copy-paste errors) and results in a confusing error message from deep within the typing module rather than from FastAPI.

## Fix

The function should validate that string annotations are valid Python expressions before creating a `ForwardRef`, or catch the `SyntaxError` and provide a more helpful error message:

```diff
--- a/fastapi/dependencies/utils.py
+++ b/fastapi/dependencies/utils.py
@@ -246,8 +246,14 @@ def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:

 def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
     if isinstance(annotation, str):
-        annotation = ForwardRef(annotation)
-        annotation = evaluate_forwardref(annotation, globalns, globalns)
+        try:
+            annotation = ForwardRef(annotation)
+            annotation = evaluate_forwardref(annotation, globalns, globalns)
+        except SyntaxError as e:
+            raise ValueError(
+                f"Invalid type annotation {annotation!r}: {e}. "
+                "String annotations must be valid Python expressions."
+            ) from e
     return annotation
```