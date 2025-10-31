# Bug Report: Cython.Build.Dependencies resolve_depend Crashes on Empty String

**Target**: `Cython.Build.Dependencies.resolve_depend`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `resolve_depend()` function crashes with `IndexError` when passed an empty string as the `depend` parameter, due to unchecked indexing with `depend[0]` and `depend[-1]`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Build.Dependencies import resolve_depend
import pytest


@given(st.lists(st.text(min_size=1)))
def test_resolve_depend_handles_all_strings(include_dirs):
    try:
        result = resolve_depend("", tuple(include_dirs))
    except IndexError:
        pytest.fail("resolve_depend should not raise IndexError on empty string")
```

**Failing input**: `depend=""` with any `include_dirs` tuple

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import resolve_depend

result = resolve_depend("", ())
```

Output:
```
IndexError: string index out of range
```

## Why This Is A Bug

The function checks `depend[0]` and `depend[-1]` without first verifying that `depend` is non-empty. While empty dependency strings may be rare in practice, robust code should handle edge cases gracefully rather than crashing. The function should either validate input or handle empty strings appropriately.

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -450,7 +450,7 @@ def resolve_depends(depends, include_dirs):

 @cached_function
 def resolve_depend(depend, include_dirs):
-    if depend[0] == '<' and depend[-1] == '>':
+    if depend and depend[0] == '<' and depend[-1] == '>':
         return None
     for dir in include_dirs:
         path = join_path(dir, depend)
```