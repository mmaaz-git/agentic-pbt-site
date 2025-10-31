# Bug Report: dask.utils.key_split Raises Exceptions Instead of Returning 'Other'

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function has exception handling that should return 'Other' for invalid inputs, but it raises TypeError for unhashable types (due to @lru_cache) and UnicodeDecodeError for invalid UTF-8 bytes before the try-except block can handle them.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import key_split

@given(st.one_of(
    st.text(),
    st.binary(),
    st.tuples(st.text()),
    st.none(),
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers()),
))
def test_key_split_never_raises(key):
    result = key_split(key)
    assert isinstance(result, str)
```

**Failing inputs**:
- `key = []` → TypeError: unhashable type: 'list'
- `key = b'\x80'` → UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80

## Reproducing the Bug

```python
from dask.utils import key_split

try:
    result = key_split(b'\x80')
    print(f"key_split(b'\\x80') = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

try:
    result = key_split([])
    print(f"key_split([]) = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")
```

Output:
```
BUG: Raised UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
BUG: Raised TypeError: unhashable type: 'list'
```

## Why This Is A Bug

The function has exception handling code that catches `Exception` and returns 'Other':

```python
try:
    words = s.split("-")
    ...
except Exception:
    return "Other"
```

However, this exception handling is bypassed by:

1. **Invalid UTF-8 bytes**: The bytes handling code at line 1978-1979 calls `s.decode()` which can raise `UnicodeDecodeError` before reaching the try-except block.

2. **Unhashable types**: The `@functools.lru_cache(100000)` decorator at line 1947 requires hashable inputs. Lists, dicts, and sets raise `TypeError: unhashable type` before the function body executes.

This is problematic because:
- The docstring examples show bytes as valid inputs (line 1964-1965)
- The function has explicit exception handling suggesting it should handle all inputs gracefully
- Dask task keys might include various types that should degrade gracefully

## Fix

```diff
 @functools.lru_cache(100000)
 def key_split(s):
     """
     >>> key_split('x')
     'x'
     ...
     """
-    # If we convert the key, recurse to utilize LRU cache better
-    if type(s) is bytes:
-        return key_split(s.decode())
-    if type(s) is tuple:
-        return key_split(s[0])
     try:
+        # If we convert the key, recurse to utilize LRU cache better
+        if type(s) is bytes:
+            return key_split(s.decode())
+        if type(s) is tuple:
+            return key_split(s[0])
+
         words = s.split("-")
         if not words[0][0].isalpha():
             result = words[0].split(",")[0].strip("_'()\"")
         else:
             result = words[0]
         for word in words[1:]:
             if word.isalpha() and not (
                 len(word) == 8 and hex_pattern.match(word) is not None
             ):
                 result += "-" + word
             else:
                 break
         if len(result) == 32 and re.match(r"[a-f0-9]{32}", result):
             return "data"
         else:
             if result[0] == "<":
                 result = result.strip("<>").split()[0].split(".")[-1]
             return sys.intern(result)
-    except Exception:
+    except (Exception, TypeError):
+        # TypeError can be raised by lru_cache for unhashable types
         return "Other"
```

Note: This fix still won't handle unhashable types because the `@lru_cache` decorator raises before the function executes. A more complete fix would be:

```diff
+def _key_split_impl(s):
+    """Implementation of key_split without caching."""
+    try:
+        if type(s) is bytes:
+            return _key_split_impl(s.decode())
+        if type(s) is tuple:
+            return _key_split_impl(s[0])
+
+        words = s.split("-")
+        ...
+        return sys.intern(result)
+    except Exception:
+        return "Other"
+
+@functools.lru_cache(100000)
+def _key_split_cached(s):
+    """Cached wrapper that only accepts hashable types."""
+    return _key_split_impl(s)
+
 def key_split(s):
-    """..."""
-    # If we convert the key, recurse to utilize LRU cache better
-    if type(s) is bytes:
-        return key_split(s.decode())
-    if type(s) is tuple:
-        return key_split(s[0])
-    try:
-        ...
-    except Exception:
-        return "Other"
+    """..."""
+    try:
+        # Only use cache for hashable types
+        hash(s)
+        return _key_split_cached(s)
+    except TypeError:
+        # Unhashable type, use uncached implementation
+        return _key_split_impl(s)
```