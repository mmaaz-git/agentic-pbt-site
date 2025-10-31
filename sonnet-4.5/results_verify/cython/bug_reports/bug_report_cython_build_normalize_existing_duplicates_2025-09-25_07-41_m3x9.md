# Bug Report: Cython.Build.Dependencies.normalize_existing Produces Duplicate Paths

**Target**: `Cython.Build.Dependencies.normalize_existing`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`normalize_existing` returns duplicate paths when given both a relative path and the corresponding absolute path to the same file.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Build.Dependencies import normalize_existing
import tempfile
import os

valid_filename = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='\x00/'),
    min_size=1,
    max_size=20
).filter(lambda s: '{' not in s and '}' not in s and ',' not in s and '\x00' not in s and s not in ['.', '..'])


@given(valid_filename)
def test_normalize_existing_no_duplicates(filename):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_file = os.path.join(tmpdir, 'base.txt')
        with open(base_file, 'w') as f:
            f.write('base')

        existing_file = os.path.join(tmpdir, filename)
        with open(existing_file, 'w') as f:
            f.write('test')

        paths = [filename, existing_file]

        normalized, base_dir = normalize_existing(base_file, paths)

        assert len(normalized) == len(set(normalized)), "Should not produce duplicates"
```

**Failing input**: `filename='0'` (or any valid filename)

## Reproducing the Bug

```python
from Cython.Build.Dependencies import normalize_existing
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    base_file = os.path.join(tmpdir, 'base.txt')
    with open(base_file, 'w') as f:
        f.write('base')

    existing_file = os.path.join(tmpdir, 'test.txt')
    with open(existing_file, 'w') as f:
        f.write('test')

    paths = ['test.txt', existing_file]

    normalized, base_dir = normalize_existing(base_file, paths)

    print(f'Input: {paths}')
    print(f'Output: {normalized}')
    print(f'Unique: {set(normalized)}')
```

Output:
```
Input: ['test.txt', '/tmp/tmpxxx/test.txt']
Output: ['/tmp/tmpxxx/test.txt', '/tmp/tmpxxx/test.txt']
Unique: {'/tmp/tmpxxx/test.txt'}
```

## Why This Is A Bug

1. **The function name suggests normalization**: Normalization typically means converting different representations of the same entity to a canonical form, which implies deduplication
2. **Input deduplication is insufficient**: The function calls `set(rel_paths)` before processing, but this only removes exact string duplicates, not semantic duplicates (relative vs absolute paths to the same file)
3. **Duplicates serve no purpose**: Having the same normalized path appear multiple times in the output provides no value and could cause issues in build systems
4. **Violates expected behavior**: Path normalization functions should return each unique file path once

The root cause:
```python
def normalize_existing(base_path, rel_paths):
    return normalize_existing0(os.path.dirname(base_path), tuple(set(rel_paths)))
```

The `set(rel_paths)` removes exact duplicates, but when processing:
- `'test.txt'` (relative) → normalized to `/tmp/tmpxxx/test.txt`
- `/tmp/tmpxxx/test.txt` (absolute) → kept as `/tmp/tmpxxx/test.txt`

Both produce the same normalized path, creating a duplicate.

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -407,8 +407,11 @@ def normalize_existing0(base_dir, rel_paths):
         else:
             normalized.append(rel)
     return (normalized, needed_base)


 def normalize_existing(base_path, rel_paths):
-    return normalize_existing0(os.path.dirname(base_path), tuple(set(rel_paths)))
+    normalized, needed_base = normalize_existing0(os.path.dirname(base_path), tuple(set(rel_paths)))
+    # Remove duplicates that can occur when both relative and absolute paths to the same file are provided
+    seen = set()
+    unique_normalized = [p for p in normalized if p not in seen and not seen.add(p)]
+    return (unique_normalized, needed_base)
```

This fix deduplicates the normalized paths after processing, ensuring each unique file path appears exactly once in the output.