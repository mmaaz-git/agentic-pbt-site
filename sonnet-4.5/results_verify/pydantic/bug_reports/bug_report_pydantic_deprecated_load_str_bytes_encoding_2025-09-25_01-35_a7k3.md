# Bug Report: pydantic.deprecated.parse.load_str_bytes Ignores encoding Parameter for Pickle

**Target**: `pydantic.deprecated.parse.load_str_bytes`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `load_str_bytes` function ignores the `encoding` parameter when converting a string to bytes for pickle deserialization, always using UTF-8 encoding instead.

## Property-Based Test

```python
import pickle
from hypothesis import given, strategies as st
from pydantic.deprecated.parse import load_str_bytes, Protocol

@given(st.text(min_size=1, max_size=100))
def test_load_str_bytes_encoding_consistency(data):
    pickled_bytes = pickle.dumps(data)

    for encoding in ['utf-8', 'latin-1', 'ascii']:
        try:
            encoded_str = pickled_bytes.decode(encoding)
            result = load_str_bytes(
                encoded_str,
                proto=Protocol.pickle,
                allow_pickle=True,
                encoding=encoding
            )
            assert result == data
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
```

**Failing input**: Pickle data decoded with 'latin-1' but function uses 'utf-8' to re-encode

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

import pickle
from pydantic.deprecated.parse import load_str_bytes, Protocol

data = "test"
pickled_bytes = pickle.dumps(data)
latin1_str = pickled_bytes.decode('latin-1')

result = load_str_bytes(
    latin1_str,
    proto=Protocol.pickle,
    allow_pickle=True,
    encoding='latin-1'
)
```

Output:
```
UnpicklingError: invalid load key, '\xc2'.
```

The function fails because it re-encodes the string using UTF-8 (the default) instead of 'latin-1' as specified.

## Why This Is A Bug

The function accepts an `encoding` parameter, documenting that it should be used when converting between strings and bytes. However, the implementation only uses this parameter when decoding bytes to string for JSON (line 48), but not when encoding string to bytes for pickle (line 53).

This violates the principle of least surprise - when a user specifies an encoding parameter, they expect it to be used consistently throughout the function.

## Fix

The fix is simple: use the `encoding` parameter when calling `encode()`:

```diff
--- a/pydantic/deprecated/parse.py
+++ b/pydantic/deprecated/parse.py
@@ -50,7 +50,7 @@ def load_str_bytes(
     elif proto == Protocol.pickle:
         if not allow_pickle:
             raise RuntimeError('Trying to decode with pickle with allow_pickle=False')
-        bb = b if isinstance(b, bytes) else b.encode()  # type: ignore
+        bb = b if isinstance(b, bytes) else b.encode(encoding)  # type: ignore
         return pickle.loads(bb)
     else:
         raise TypeError(f'Unknown protocol: {proto}')
```