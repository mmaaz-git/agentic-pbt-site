# Bug Report: extended_iglob Brace Expansion Duplicates

**Target**: `Cython.Build.Dependencies.extended_iglob`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `extended_iglob` function returns duplicate file paths when brace expansion patterns contain duplicate alternatives (e.g., `{py,pyx,py}`). This violates the implicit no-duplicates guarantee that the function provides for recursive globs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import os
import tempfile
from pathlib import Path
from Cython.Build.Dependencies import extended_iglob


def test_brace_expansion_no_duplicates():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        (test_dir / 'test.py').write_text('test')
        (test_dir / 'test.pyx').write_text('test')

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            pattern = 'test.{py,pyx,py}'
            results = list(extended_iglob(pattern))

            unique_results = list(set(results))
            assert len(results) == len(unique_results), \
                f"Duplicate file in brace expansion: {results} vs {unique_results}"

        finally:
            os.chdir(orig_cwd)
```

**Failing input**: `'test.{py,pyx,py}'`

## Reproducing the Bug

```python
import os
import tempfile
from pathlib import Path
from Cython.Build.Dependencies import extended_iglob

with tempfile.TemporaryDirectory() as tmpdir:
    test_dir = Path(tmpdir)
    (test_dir / 'test.py').write_text('test')
    (test_dir / 'test.pyx').write_text('test')

    orig_cwd = os.getcwd()
    try:
        os.chdir(tmpdir)
        pattern = 'test.{py,pyx,py}'
        results = list(extended_iglob(pattern))
        print(f"Results: {results}")
    finally:
        os.chdir(orig_cwd)
```

**Output**:
```
Results: ['test.py', 'test.pyx', 'test.py']
```

**Expected output**:
```
Results: ['test.py', 'test.pyx']
```

## Why This Is A Bug

1. The function already implements duplicate prevention for recursive globs (`**/`) using a `seen` set (lines 54-68).
2. The inconsistency between brace expansion and recursive glob behavior is unexpected.
3. Duplicates cause:
   - Wasted processing time (files processed multiple times)
   - Potential compilation errors in Cython's build system
   - Violations of caller expectations about unique results

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -40,13 +40,17 @@ def _make_relative(file_paths, base=None):

 def extended_iglob(pattern):
+    seen = set()
     if '{' in pattern:
         m = re.match('(.*){([^}]+)}(.*)', pattern)
         if m:
             before, switch, after = m.groups()
             for case in switch.split(','):
                 for path in extended_iglob(before + case + after):
-                    yield path
+                    if path not in seen:
+                        seen.add(path)
+                        yield path
             return

     # We always accept '/' and also '\' on Windows,
     # because '/' is generally common for relative paths.
     if '**/' in pattern or os.sep == '\\' and '**\\' in pattern:
-        seen = set()
         first, rest = re.split(r'\*\*[%s]' % ('/\\\\' if os.sep == '\\' else '/'), pattern, 1)
         if first:
             first = iglob(first + os.sep)
```

The fix moves the `seen` set to the top of the function and applies it to brace expansion as well as recursive globs, ensuring all paths returned are unique regardless of which feature is used.