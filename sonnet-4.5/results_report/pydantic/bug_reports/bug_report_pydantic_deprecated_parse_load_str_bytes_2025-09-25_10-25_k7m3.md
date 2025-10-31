# Bug Report: pydantic.deprecated.parse.load_str_bytes Encoding Parameter Ignored for Pickle Protocol

**Target**: `pydantic.deprecated.parse.load_str_bytes`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `load_str_bytes` function ignores the `encoding` parameter when converting string input to bytes for pickle protocol, causing unpickling failures when the string contains non-ASCII characters.

## Property-Based Test

```python
import pickle
import warnings
from hypothesis import given, settings, strategies as st
from pydantic.deprecated.parse import load_str_bytes, Protocol

warnings.filterwarnings('ignore', category=DeprecationWarning)

@settings(max_examples=500)
@given(st.lists(st.integers()))
def test_load_str_bytes_pickle_encoding_parameter(lst):
    pickled_bytes = pickle.dumps(lst)
    pickled_str = pickled_bytes.decode('latin1')

    result = load_str_bytes(pickled_str, proto=Protocol.pickle,
                          encoding='latin1', allow_pickle=True)
    assert result == lst
```

**Failing input**: `lst=[]`

## Reproducing the Bug

```python
import pickle
from pydantic.deprecated.parse import load_str_bytes, Protocol

data = []
pickled_bytes = pickle.dumps(data)
pickled_str = pickled_bytes.decode('latin1')

load_str_bytes(pickled_str, proto=Protocol.pickle,
               encoding='latin1', allow_pickle=True)
```

Output:
```
_pickle.UnpicklingError: invalid load key, '\xc2'.
```

## Why This Is A Bug

The function signature accepts `b: str | bytes` and has an `encoding` parameter. When pickle protocol is used with string input, the code does `b.encode()` which uses UTF-8 by default, ignoring the `encoding` parameter. This causes corruption when the string contains non-ASCII bytes (which pickle data always does).

For consistency with the JSON protocol handling (which respects `encoding` when decoding bytes), the pickle protocol should also use the `encoding` parameter when encoding strings.

## Fix

```diff
--- a/pydantic/deprecated/parse.py
+++ b/pydantic/deprecated/parse.py
@@ -51,7 +51,7 @@ def load_str_bytes(
     elif proto == Protocol.pickle:
         if not allow_pickle:
             raise RuntimeError('Trying to decode with pickle with allow_pickle=False')
-        bb = b if isinstance(b, bytes) else b.encode()  # type: ignore
+        bb = b if isinstance(b, bytes) else b.encode(encoding)  # type: ignore
         return pickle.loads(bb)
     else:
         raise TypeError(f'Unknown protocol: {proto}')
```