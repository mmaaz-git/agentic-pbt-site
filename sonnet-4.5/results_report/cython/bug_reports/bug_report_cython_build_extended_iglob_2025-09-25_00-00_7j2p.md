# Bug Report: Cython.Build.Dependencies.extended_iglob Returns Duplicates

**Target**: `Cython.Build.Dependencies.extended_iglob`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `extended_iglob` function returns duplicate file paths when brace expansion patterns contain repeated alternatives (e.g., `{a,a}.txt`), inconsistent with the `**/` code path which explicitly deduplicates results.

## Property-Based Test

```python
import os
import tempfile
from hypothesis import given, settings, strategies as st
from Cython.Build.Dependencies import extended_iglob

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=10), min_size=1, max_size=5))
@settings(max_examples=200)
def test_extended_iglob_no_duplicates(alternatives):
    with tempfile.TemporaryDirectory() as tmpdir:
        for alt in alternatives:
            filepath = os.path.join(tmpdir, f"{alt}.txt")
            with open(filepath, 'w') as f:
                f.write('')

        pattern = os.path.join(tmpdir, '{' + ','.join(alternatives) + '}.txt')
        results = list(extended_iglob(pattern))

        assert len(results) == len(set(results)), \
            f"Found duplicates in extended_iglob results: {results}"
```

**Failing input**: `alternatives=['a', 'a']`

## Reproducing the Bug

```python
import os
import tempfile
from Cython.Build.Dependencies import extended_iglob

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test.txt")
    with open(filepath, 'w') as f:
        f.write('')

    pattern = os.path.join(tmpdir, '{test,test}.txt')
    results = list(extended_iglob(pattern))

    print(f"Pattern: {pattern}")
    print(f"Results: {results}")
    assert len(results) == len(set(results)), f"Expected unique results, got duplicates: {results}"
```

## Why This Is A Bug

The `extended_iglob` function has inconsistent behavior between its brace expansion code path (lines 42-49) and its `**/` recursive glob code path (lines 53-68). The `**/` path explicitly uses a `seen` set to prevent duplicates, but the brace expansion path blindly yields all results without deduplication. This means patterns like `{a,a,a}.txt` will yield the same file three times. While users might not intentionally write such patterns, they can easily arise from programmatically generated patterns.

## Fix

Add duplicate checking to the brace expansion code path, consistent with the `**/` path:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -40,11 +40,14 @@ def _make_relative(file_paths, base=None):

 def extended_iglob(pattern):
     if '{' in pattern:
         m = re.match('(.*){([^}]+)}(.*)', pattern)
         if m:
             before, switch, after = m.groups()
+            seen = set()
             for case in switch.split(','):
                 for path in extended_iglob(before + case + after):
-                    yield path
+                    if path not in seen:
+                        seen.add(path)
+                        yield path
             return
```