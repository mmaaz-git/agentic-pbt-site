# Bug Report: Cython.Build.Dependencies resolve_depend Crashes on Empty String

**Target**: `Cython.Build.Dependencies.resolve_depend`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `resolve_depend()` function crashes with `IndexError` when passed an empty string as the `depend` parameter, due to unchecked indexing operations `depend[0]` and `depend[-1]`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Build.Dependencies import resolve_depend


@given(st.lists(st.text(min_size=1)), st.text())
def test_resolve_depend_handles_all_depend_strings(include_dirs, depend):
    result = resolve_depend(depend, tuple(include_dirs))
    assert result is None or isinstance(result, str)
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
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".../Cython/Build/Dependencies.py", line 453, in resolve_depend
    if depend[0] == '<' and depend[-1] == '>':
IndexError: string index out of range
```

## Why This Is A Bug

The function unconditionally accesses `depend[0]` and `depend[-1]` without checking if the string is empty. While empty dependency strings may not occur in typical usage, defensive programming dictates that edge cases should be handled gracefully. A crash on unexpected input makes the build system fragile.

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