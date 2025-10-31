# Bug Report: generate_operation_id_for_path Incomplete Special Character Replacement

**Target**: `fastapi.openapi.utils.generate_operation_id_for_path`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `generate_operation_id_for_path` function inconsistently handles special character replacement. It replaces non-word characters with underscores in the `name` and `path` parameters, but fails to do so for the `method` parameter, allowing special characters to appear in the final operation ID.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from fastapi.openapi.utils import generate_operation_id_for_path
import re

@given(
    name=st.text(min_size=1, max_size=50),
    path=st.text(min_size=1, max_size=50),
    method=st.text(min_size=1, max_size=10),
)
@settings(suppress_health_check=[])
def test_generate_operation_id_for_path_only_word_chars_and_underscores(name, path, method):
    result = generate_operation_id_for_path(name=name, path=path, method=method)
    assert re.match(r'^[\w]+$', result), f"Result '{result}' contains non-word characters"
```

**Failing input**: `name='0', path='0', method=':'`

## Reproducing the Bug

```python
from fastapi.openapi.utils import generate_operation_id_for_path

result = generate_operation_id_for_path(name='0', path='0', method=':')
print(result)

assert ':' not in result
```

**Output**:
```
00_:
AssertionError
```

**Additional example**:
```python
result = generate_operation_id_for_path(name='get_user', path='/api/users', method='POST@')
print(result)
```

**Output**: `get_user_api_users_post@` (contains `@`)

## Why This Is A Bug

The function applies `re.sub(r"\W", "_", operation_id)` to clean the concatenation of `name` and `path`, but then directly appends `_{method.lower()}` without applying the same cleaning to the `method` parameter. This creates inconsistent behavior where special characters are removed from some inputs but not others.

While HTTP methods in typical usage (GET, POST, etc.) don't contain special characters, the function accepts arbitrary strings and should handle them consistently. The incomplete sanitization violates the established pattern and could lead to operation IDs with unexpected special characters.

## Fix

```diff
def generate_operation_id_for_path(
    *, name: str, path: str, method: str
) -> str:  # pragma: nocover
    warnings.warn(
        "fastapi.utils.generate_operation_id_for_path() was deprecated, "
        "it is not used internally, and will be removed soon",
        DeprecationWarning,
        stacklevel=2,
    )
    operation_id = f"{name}{path}"
    operation_id = re.sub(r"\W", "_", operation_id)
-   operation_id = f"{operation_id}_{method.lower()}"
+   method_cleaned = re.sub(r"\W", "_", method.lower())
+   operation_id = f"{operation_id}_{method_cleaned}"
    return operation_id
```