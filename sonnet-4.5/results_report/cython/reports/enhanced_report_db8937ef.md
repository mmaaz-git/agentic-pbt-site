# Bug Report: Cython.Build.Dependencies.extended_iglob Returns Duplicate File Paths

**Target**: `Cython.Build.Dependencies.extended_iglob`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `extended_iglob` function returns duplicate file paths when brace expansion patterns contain repeated alternatives (e.g., `{a,a}.txt`), which is inconsistent with the function's `**/` recursive glob code path that explicitly prevents duplicates.

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

if __name__ == "__main__":
    test_extended_iglob_no_duplicates()
```

<details>

<summary>
**Failing input**: `alternatives=['a', 'a']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 22, in <module>
    test_extended_iglob_no_duplicates()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 7, in test_extended_iglob_no_duplicates
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 18, in test_extended_iglob_no_duplicates
    assert len(results) == len(set(results)), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Found duplicates in extended_iglob results: ['/tmp/tmpffm2b0e8/a.txt', '/tmp/tmpffm2b0e8/a.txt']
Falsifying example: test_extended_iglob_no_duplicates(
    alternatives=['a', 'a'],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/59/hypo.py:19
```
</details>

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
    print(f"Number of results: {len(results)}")
    print(f"Number of unique results: {len(set(results))}")

    if len(results) != len(set(results)):
        print(f"ERROR: Expected unique results, got duplicates")
        print(f"Duplicate entries found!")
    else:
        print(f"OK: All results are unique")
```

<details>

<summary>
Duplicate file paths returned for pattern with repeated alternatives
</summary>
```
Pattern: /tmp/tmplql93n4m/{test,test}.txt
Results: ['/tmp/tmplql93n4m/test.txt', '/tmp/tmplql93n4m/test.txt']
Number of results: 2
Number of unique results: 1
ERROR: Expected unique results, got duplicates
Duplicate entries found!
```
</details>

## Why This Is A Bug

The `extended_iglob` function exhibits inconsistent behavior between its two main code paths. When processing recursive glob patterns containing `**/`, the function explicitly uses a `seen` set (lines 54-68 in Dependencies.py) to track and prevent duplicate file paths from being yielded. However, when processing brace expansion patterns (lines 42-49), the function blindly yields all results without any deduplication logic, causing the same file path to be returned multiple times when the pattern contains repeated alternatives.

This inconsistency within the same function strongly suggests that duplicate prevention was the intended behavior for all code paths. The bug causes practical issues in build systems where `cythonize()` uses this function - duplicate file paths lead to the same file being compiled multiple times, resulting in wasted CPU cycles and potentially triggering errors in build systems that expect unique file lists. While users don't typically write patterns like `{a,a}.txt` intentionally, such patterns can easily arise from programmatically generated glob patterns or when combining multiple pattern sources.

## Relevant Context

The `extended_iglob` function is an internal utility in Cython that extends Python's standard `iglob` with additional pattern support, particularly brace expansion which is not part of standard Python glob. The function is primarily used by `cythonize()` to process file patterns for compilation.

The brace expansion feature allows patterns like `{foo,bar,baz}.pyx` to expand to `foo.pyx`, `bar.pyx`, and `baz.pyx`. This is a shell-style globbing feature that Cython added for convenience.

Code location: `/Cython/Build/Dependencies.py`, lines 41-71
Documentation: The function lacks a docstring and is not documented in the official Cython documentation, though `cythonize()` documentation mentions it accepts glob patterns.

## Proposed Fix

Add duplicate checking to the brace expansion code path to maintain consistency with the recursive glob path:

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