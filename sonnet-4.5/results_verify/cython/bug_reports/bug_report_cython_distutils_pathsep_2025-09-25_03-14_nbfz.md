# Bug Report: Cython.Distutils Empty Strings in Include Paths

**Target**: `Cython.Distutils.build_ext.finalize_options`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `finalize_options` method creates empty strings in `cython_include_dirs` when the input string contains consecutive path separators, causing unintended inclusion of the current directory.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@settings(max_examples=500)
@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=5),
    st.integers(min_value=2, max_value=5)
)
def test_empty_strings_from_pathsep(paths, num_consecutive_seps):
    """
    Property: Include directories list should not contain empty strings
    """
    dist = Distribution()
    cmd = build_ext(dist)
    cmd.initialize_options()

    path_string = (os.pathsep * num_consecutive_seps).join(paths)
    cmd.cython_include_dirs = path_string
    cmd.finalize_options()

    result = cmd.cython_include_dirs
    assert '' not in result, f"Empty strings found in: {result} from input {repr(path_string)}"
```

**Failing input**: `paths=['0', '0'], num_consecutive_seps=2` (produces string `'0::0'`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import os
from distutils.dist import Distribution
from Cython.Distutils import build_ext

dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()

cmd.cython_include_dirs = "/usr/include::/usr/local/include"
cmd.finalize_options()

print(cmd.cython_include_dirs)
```

**Output**: `['/usr/include', '', '/usr/local/include']`

## Why This Is A Bug

When path strings contain consecutive separators (e.g., `::` on Unix or `;;` on Windows), the `split()` method creates empty strings in the resulting list. In filesystem path contexts, an empty string typically resolves to the current directory, which means:

1. The current directory gets unintentionally added to include paths
2. This can cause unexpected header file conflicts
3. It creates a security risk if untrusted files exist in the current directory
4. The behavior is silent with no warning to the user

This violates the expected behavior that only explicitly specified directories should be included.

## Fix

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -72,7 +72,7 @@ class build_ext(_build_ext):
         if self.cython_include_dirs is None:
             self.cython_include_dirs = []
         elif isinstance(self.cython_include_dirs, str):
-            self.cython_include_dirs = \
-                self.cython_include_dirs.split(os.pathsep)
+            self.cython_include_dirs = [
+                d for d in self.cython_include_dirs.split(os.pathsep) if d
+            ]
         if self.cython_directives is None:
```