# Bug Report: fastapi.dependencies.utils - Crash on Keyword String Type Annotations

**Target**: `fastapi.dependencies.utils.get_typed_annotation`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

FastAPI crashes with a `SyntaxError` when processing endpoint functions that use Python keywords (like "if", "class", "def") as string type annotations, even though such annotations are syntactically valid in Python.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from fastapi.dependencies.utils import get_typed_annotation
import keyword


@settings(max_examples=200)
@given(
    st.sampled_from(keyword.kwlist),  # Use Python's list of keywords directly
)
def test_get_typed_annotation_handles_keywords(type_str):
    try:
        result = get_typed_annotation(type_str, {})
    except (NameError, AttributeError):
        pass
    except SyntaxError:
        assert False, f"SyntaxError should not be raised for valid identifier '{type_str}'"

# Run the test
if __name__ == "__main__":
    test_get_typed_annotation_handles_keywords()
```

<details>

<summary>
**Failing input**: `'and'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/typing.py", line 1035, in __init__
    code = compile(arg_to_compile, '<string>', 'eval')
  File "<string>", line 1
    and
    ^^^
SyntaxError: invalid syntax

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 12, in test_get_typed_annotation_handles_keywords
    result = get_typed_annotation(type_str, {})
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 249, in get_typed_annotation
    annotation = ForwardRef(annotation)
  File "/home/npc/miniconda/lib/python3.13/typing.py", line 1037, in __init__
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
SyntaxError: Forward reference must be an expression -- got 'and'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 20, in <module>
    test_get_typed_annotation_handles_keywords()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 7, in test_get_typed_annotation_handles_keywords
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 16, in test_get_typed_annotation_handles_keywords
    assert False, f"SyntaxError should not be raised for valid identifier '{type_str}'"
           ^^^^^
AssertionError: SyntaxError should not be raised for valid identifier 'and'
Falsifying example: test_get_typed_annotation_handles_keywords(
    type_str='and',
)
```
</details>

## Reproducing the Bug

```python
from fastapi.dependencies.utils import get_dependant

# Test case: Using Python keyword "if" as a string type annotation
def endpoint_function(x: "if"):
    """Function with a keyword as string type annotation"""
    return {"x": x}

# Try to process this function through FastAPI's dependency system
try:
    dependant = get_dependant(path="/test", call=endpoint_function)
    print("SUCCESS: Function processed without error")
    print(f"Dependant: {dependant}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
SyntaxError crash when processing endpoint function
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/typing.py", line 1035, in __init__
    code = compile(arg_to_compile, '<string>', 'eval')
  File "<string>", line 1
    if
    ^^
SyntaxError: invalid syntax

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/repo.py", line 10, in <module>
    dependant = get_dependant(path="/test", call=endpoint_function)
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 274, in get_dependant
    endpoint_signature = get_typed_signature(call)
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 239, in get_typed_signature
    annotation=get_typed_annotation(param.annotation, globalns),
               ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 249, in get_typed_annotation
    annotation = ForwardRef(annotation)
  File "/home/npc/miniconda/lib/python3.13/typing.py", line 1037, in __init__
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
SyntaxError: Forward reference must be an expression -- got 'if'
ERROR: SyntaxError: Forward reference must be an expression -- got 'if'
```
</details>

## Why This Is A Bug

This is a bug because FastAPI crashes on syntactically valid Python code. While using Python keywords as string type annotations is extremely unusual and not practical, it is technically allowed by Python's syntax:

1. **Valid Python Syntax**: The function `def foo(x: "if"): pass` is syntactically valid Python code that can be defined, inspected, and called without issues in standard Python.

2. **Unexpected Crash**: FastAPI's dependency injection system crashes when processing such functions, even though the Python interpreter accepts them.

3. **Assumption Mismatch**: FastAPI assumes all string annotations are forward references to actual types that can be resolved. Python's `ForwardRef` class enforces that the string must be a valid Python expression (excluding keywords), but this restriction is not inherent to Python's annotation system itself.

4. **Poor Error Experience**: The crash happens at runtime during dependency resolution with an unclear error message that doesn't help users understand what went wrong or how to fix it.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py` at line 249 in the `get_typed_annotation` function:

```python
def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)  # <-- Crashes here on keywords
        annotation = evaluate_forwardref(annotation, globalns, globalns)
    return annotation
```

The Python `ForwardRef` class (from `typing` module) validates that the string is a valid Python expression by attempting to compile it with `compile(arg, '<string>', 'eval')`. Keywords cannot be used in expression context, causing the compilation to fail.

FastAPI documentation: https://fastapi.tiangolo.com/
Python typing documentation: https://docs.python.org/3/library/typing.html

## Proposed Fix

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