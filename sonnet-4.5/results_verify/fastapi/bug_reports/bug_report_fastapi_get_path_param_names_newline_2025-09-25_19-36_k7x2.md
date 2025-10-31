# Bug Report: fastapi.dependencies.utils.get_path_param_names Inconsistent Newline Handling

**Target**: `fastapi.dependencies.utils.get_path_param_names`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_path_param_names` function uses a regex pattern that inconsistently handles whitespace characters in path parameter names. Specifically, it extracts spaces and tabs but fails to extract newlines, due to the regex pattern `{(.*?)}` where `.` doesn't match newlines by default.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.dependencies.utils import get_path_param_names

@given(st.lists(st.text(alphabet=st.characters(blacklist_characters='{}'), min_size=1)))
def test_get_path_param_names_extracts_params(param_names):
    path = '/' + '/'.join(['{' + name + '}' for name in param_names])
    result = get_path_param_names(path)
    assert result == set(param_names)
```

**Failing input**: `['\n']`

## Reproducing the Bug

```python
from fastapi.dependencies.utils import get_path_param_names

path_with_space = "/{ }"
path_with_tab = "/{\t}"
path_with_newline = "/{\n}"

print(f"Space: {get_path_param_names(path_with_space)}")
print(f"Tab: {get_path_param_names(path_with_tab)}")
print(f"Newline: {get_path_param_names(path_with_newline)}")
```

Output:
```
Space: {' '}
Tab: {'\t'}
Newline: set()
```

## Why This Is A Bug

The function exhibits inconsistent behavior when extracting parameter names containing different types of whitespace. While spaces and tabs are successfully extracted, newlines are silently ignored. This inconsistency stems from the regex pattern `{(.*?)}` where the `.` metacharacter doesn't match newline characters by default in Python's `re` module.

This creates a situation where FastAPI accepts path definitions with newlines but fails to recognize them as path parameters, potentially causing confusing runtime behavior.

## Fix

```diff
--- a/fastapi/dependencies/utils.py
+++ b/fastapi/dependencies/utils.py
@@ -1,4 +1,4 @@
 def get_path_param_names(path: str) -> Set[str]:
-    return set(re.findall("{(.*?)}", path))
+    return set(re.findall("{([^}]*)}", path))
```

The fix changes the regex pattern from `{(.*?)}` to `{([^}]*)}`, which matches any character except the closing brace `}`. This ensures consistent extraction of all characters between braces, including newlines.