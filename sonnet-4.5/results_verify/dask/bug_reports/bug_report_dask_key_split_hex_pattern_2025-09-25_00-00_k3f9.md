# Bug Report: dask.utils.key_split Hex Pattern Does Not Match Digits

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `key_split` function fails to strip 8-character hexadecimal suffixes that contain digits (0-9), only stripping those containing letters (a-f). This is due to an incorrect regular expression pattern `[a-f]+` that excludes digits.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from dask.utils import key_split


@given(
    key_base=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    hex_suffix=st.text(min_size=8, max_size=8, alphabet='0123456789abcdef'),
)
@settings(max_examples=1000)
def test_key_split_removes_8char_hex_suffix(key_base, hex_suffix):
    key = f"{key_base}-{hex_suffix}"
    result = key_split(key)
    expected = key_base
    assert result == expected, f"key_split('{key}') = '{result}', expected '{expected}' (hex suffix '{hex_suffix}' should be stripped)"
```

**Failing input**: `key_base='task'`, `hex_suffix='12345678'` (or any 8-char hex with digits)

## Reproducing the Bug

```python
from dask.utils import key_split

assert key_split('task-abcdefab') == 'task'
assert key_split('task-12345678') == 'task'
```

The first assertion passes, but the second fails with:
```
AssertionError: assert 'task-12345678' == 'task'
```

## Why This Is A Bug

1. **Documented behavior**: The docstring shows `key_split('x-abcdefab')  # ignores hex`, indicating 8-character hex suffixes should be stripped
2. **Inconsistent implementation**: Line 1995 in utils.py correctly uses `[a-f0-9]{32}` for 32-character hashes, but line 1944 uses `[a-f]+` for 8-character hashes
3. **Hexadecimal definition**: Valid hexadecimal includes digits 0-9 and letters a-f (case-insensitive), but the pattern only matches letters
4. **Real-world impact**: Dask task keys often have hex-like suffixes (e.g., `task-12abc345`), which won't be properly normalized

## Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1941,7 +1941,7 @@ def groupby_tasks(tasks):
         yield chunk


-hex_pattern = re.compile("[a-f]+")
+hex_pattern = re.compile("[a-f0-9]+")


 @functools.lru_cache(100000)
```

This changes the pattern from `[a-f]+` (only lowercase letters a-f) to `[a-f0-9]+` (all valid hexadecimal characters), making it consistent with the 32-character hash pattern on line 1995.