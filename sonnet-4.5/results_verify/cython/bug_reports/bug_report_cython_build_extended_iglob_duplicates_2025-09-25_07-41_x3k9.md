# Bug Report: Cython.Build.Dependencies.extended_iglob Produces Duplicate Results

**Target**: `Cython.Build.Dependencies.extended_iglob`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`extended_iglob` produces duplicate file paths when brace expansion contains glob wildcard characters (like `?` or `*`) that match other items in the expansion.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from Cython.Build.Dependencies import extended_iglob
import tempfile
import os

valid_filename = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='\x00/'),
    min_size=1,
    max_size=20
).filter(lambda s: '{' not in s and '}' not in s and ',' not in s and '\x00' not in s and s not in ['.', '..'])


@given(st.lists(valid_filename, min_size=2, max_size=5))
def test_brace_expansion_no_duplicates(choices):
    assume(len(set(choices)) == len(choices))

    with tempfile.TemporaryDirectory() as tmpdir:
        for choice in choices:
            path = os.path.join(tmpdir, choice)
            os.makedirs(path, exist_ok=True)

        pattern = os.path.join(tmpdir, '{' + ','.join(choices) + '}')
        result = list(extended_iglob(pattern))

        assert len(result) == len(set(result)), "Brace expansion should not produce duplicates"
```

**Failing input**: `choices=['0', '?']`

## Reproducing the Bug

```python
from Cython.Build.Dependencies import extended_iglob
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    os.makedirs(os.path.join(tmpdir, 'a'), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, '?'), exist_ok=True)

    pattern = os.path.join(tmpdir, '{a,?}')
    result = list(extended_iglob(pattern))

    print(f'Pattern: {pattern}')
    print(f'Result: {result}')
    print(f'Expected: 2 unique paths')
    print(f'Got: {len(result)} paths ({len(set(result))} unique)')
```

Output:
```
Pattern: /tmp/tmpxxx/{a,?}
Result: ['/tmp/tmpxxx/a', '/tmp/tmpxxx/?', '/tmp/tmpxxx/a']
Expected: 2 unique paths
Got: 3 paths (2 unique)
```

## Why This Is A Bug

1. **Inconsistent with standard brace expansion**: In shell brace expansion, `{a,?}` creates two literal strings `a` and `?`, not glob patterns
2. **Inconsistent within the function**: The function uses a `seen` set to prevent duplicates for `**/` patterns (lines 13-28) but not for brace expansion (lines 2-9)
3. **Violates expected behavior**: File globbing functions should return each matching path exactly once
4. **Causes unexpected results**: When brace items contain glob characters, they're treated as patterns that can match other brace items, creating duplicates

The root cause: After splitting brace expansion `{a,?}` into `['a', '?']`, each item is passed to `iglob()` which treats `?` as a wildcard matching any single character, including `a`.

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -40,11 +40,14 @@ def join_path(path, *paths):

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
```

This fix adds duplicate detection to brace expansion, matching the pattern used for `**/` glob handling elsewhere in the same function.