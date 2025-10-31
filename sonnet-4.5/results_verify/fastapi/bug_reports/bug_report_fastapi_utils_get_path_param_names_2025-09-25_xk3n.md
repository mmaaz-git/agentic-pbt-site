# Bug Report: FastAPI get_path_param_names Empty Parameter Names

**Target**: `fastapi.utils.get_path_param_names`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_path_param_names` function incorrectly accepts and returns empty strings when paths contain empty braces `{}`, violating the semantic requirement that path parameters must have non-empty names.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.utils import get_path_param_names

@given(st.text())
def test_get_path_param_names_no_empty_strings(path):
    result = get_path_param_names(path)
    for name in result:
        assert name != '', f"Empty parameter name found for path: {path!r}"
```

**Failing input**: `"/users/{}/posts"` or any path containing `{}`

## Reproducing the Bug

```python
from fastapi.utils import get_path_param_names

result = get_path_param_names("/users/{}/posts")
print(f"Result: {result}")

assert '' in result
```

## Why This Is A Bug

Path parameters must have valid names to be referenced in function signatures and matched against request URLs. The OpenAPI specification and routing conventions require named parameters. Empty parameter names are:

1. Semantically invalid - they can't be referenced in code
2. Not matchable against Python function parameters (which can't be named "")
3. Inconsistent with the function's implied contract (Set[str] of valid parameter names)

While this doesn't cause crashes (empty strings simply never match any function parameter), it violates correctness by accepting invalid input without validation.

## Fix

```diff
 def get_path_param_names(path: str) -> Set[str]:
-    return set(re.findall("{(.*?)}", path))
+    return set(name for name in re.findall("{(.*?)}", path) if name)
```

Or alternatively, raise an error for empty parameter names:

```diff
 def get_path_param_names(path: str) -> Set[str]:
-    return set(re.findall("{(.*?)}", path))
+    param_names = set(re.findall("{(.*?)}", path))
+    if '' in param_names:
+        raise ValueError(f"Path contains empty parameter name: {path}")
+    return param_names
```