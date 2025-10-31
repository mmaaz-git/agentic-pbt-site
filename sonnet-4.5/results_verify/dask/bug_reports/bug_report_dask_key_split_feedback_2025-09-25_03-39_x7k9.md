# Bug Report: dask.utils.key_split Incorrectly Strips Legitimate English Words

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `key_split` function incorrectly strips legitimate English words that happen to be 8 characters long and contain only letters a-f (e.g., "feedback", "faceache", "beefcafe") because the hex detection pattern is too permissive.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import key_split

@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10),
       st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10))
@settings(max_examples=500)
def test_key_split_compound_key(key1, key2):
    s = f"{key1}-{key2}-1"
    result = key_split(s)

    if len(key2) != 8 or not all(c in 'abcdef' for c in key2):
        assert result == f"{key1}-{key2}"
```

**Failing input**: `key1='task'`, `key2='feedback'`

## Reproducing the Bug

```python
from dask.utils import key_split

result = key_split('task-feedback-1')
print(f"Result: {repr(result)}")
print(f"Expected: 'task-feedback'")

assert result == 'task-feedback'
```

Output:
```
Result: 'task'
Expected: 'task-feedback'
AssertionError: Bug: 'feedback' was incorrectly stripped!
```

Other affected words:
- `key_split('process-feedback-0')` returns `'process'` instead of `'process-feedback'`
- `key_split('data-faceache-1')` returns `'data'` instead of `'data-faceache'`
- `key_split('task-beefcafe-2')` returns `'task'` instead of `'task-beefcafe'`

## Why This Is A Bug

The `key_split` function is designed to extract task name prefixes by removing numeric and hex suffixes. However, it incorrectly identifies legitimate English words as hex suffixes.

The problem is in the hex detection logic (line 1990 in utils.py):

```python
if word.isalpha() and not (
    len(word) == 8 and hex_pattern.match(word) is not None
):
```

where `hex_pattern = re.compile("[a-f]+")` (line 1944).

The pattern `[a-f]+` with `match()` will match ANY word that STARTS with letters a-f, not just actual hexadecimal strings. So "feedback" (all letters are a-f) gets incorrectly classified as a hex suffix and stripped.

This violates the documented behavior shown in the docstring: `key_split('hello-world-1')` returns `'hello-world'`, preserving compound task names. Users would reasonably expect "feedback" to be preserved the same way "world" is.

## Fix

The hex pattern should be more restrictive to only match actual hexadecimal strings, not English words. Change line 1944 from:

```python
hex_pattern = re.compile("[a-f]+")
```

to:

```python
hex_pattern = re.compile(r"^[a-f0-9]{8}$")
```

And update line 1990 to use `fullmatch()` or just check if the pattern matches:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1941,7 +1941,7 @@ def parse_bytes(s):
     return result


-hex_pattern = re.compile("[a-f]+")
+hex_pattern = re.compile(r"^[a-f0-9]{8}$")


 @functools.lru_cache(100000)
@@ -1987,7 +1987,7 @@ def key_split(s):
             result = words[0]
         for word in words[1:]:
             if word.isalpha() and not (
-                len(word) == 8 and hex_pattern.match(word) is not None
+                len(word) == 8 and hex_pattern.fullmatch(word) is not None
             ):
                 result += "-" + word
             else:
```

This ensures that only strings that are EXACTLY 8 hexadecimal characters (0-9, a-f) are treated as hex suffixes, not legitimate English words like "feedback".