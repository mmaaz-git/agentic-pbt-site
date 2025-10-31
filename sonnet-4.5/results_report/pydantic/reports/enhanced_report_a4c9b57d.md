# Bug Report: pydantic.deprecated.parse.load_str_bytes Ignores Encoding Parameter for Pickle Protocol

**Target**: `pydantic.deprecated.parse.load_str_bytes`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `load_str_bytes` function ignores the `encoding` parameter when converting string input to bytes for pickle protocol, causing UnpicklingError crashes when the pickled data contains non-ASCII bytes and a non-UTF-8 encoding is specified.

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

if __name__ == "__main__":
    test_load_str_bytes_pickle_encoding_parameter()
```

<details>

<summary>
**Failing input**: `lst=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 19, in <module>
    test_load_str_bytes_pickle_encoding_parameter()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 9, in test_load_str_bytes_pickle_encoding_parameter
    @given(st.lists(st.integers()))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 14, in test_load_str_bytes_pickle_encoding_parameter
    result = load_str_bytes(pickled_str, proto=Protocol.pickle,
                          encoding='latin1', allow_pickle=True)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/deprecated/parse.py", line 54, in load_str_bytes
    return pickle.loads(bb)
           ~~~~~~~~~~~~^^^^
_pickle.UnpicklingError: invalid load key, '\xc2'.
Falsifying example: test_load_str_bytes_pickle_encoding_parameter(
    lst=[],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pickle
from pydantic.deprecated.parse import load_str_bytes, Protocol

data = []
pickled_bytes = pickle.dumps(data)
pickled_str = pickled_bytes.decode('latin1')

print(f"Original pickle bytes: {pickled_bytes}")
print(f"Decoded with latin1: {repr(pickled_str)}")

try:
    result = load_str_bytes(pickled_str, proto=Protocol.pickle,
                            encoding='latin1', allow_pickle=True)
    print(f"Successfully loaded: {result}")
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
```

<details>

<summary>
UnpicklingError: invalid load key, '\xc2'.
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/28/repo.py:12: PydanticDeprecatedSince20: `load_str_bytes` is deprecated. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  result = load_str_bytes(pickled_str, proto=Protocol.pickle,
Original pickle bytes: b'\x80\x04]\x94.'
Decoded with latin1: '\x80\x04]\x94.'
Error: invalid load key, '\xc2'.
Error type: UnpicklingError
```
</details>

## Why This Is A Bug

The `load_str_bytes` function accepts an `encoding` parameter that is intended to control string/bytes conversions. However, the implementation at line 53 of `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/deprecated/parse.py` uses `b.encode()` without passing the encoding parameter, defaulting to UTF-8.

This violates the expected behavior because:
1. The function signature explicitly includes an `encoding` parameter with a default value
2. The JSON protocol correctly uses this parameter for bytes-to-string conversion (line 48: `b.decode(encoding)`)
3. The pickle protocol ignores it for string-to-bytes conversion, creating inconsistent behavior
4. When pickle data containing bytes > 127 is decoded with a non-UTF-8 encoding (e.g., latin1) and passed as a string, the UTF-8 re-encoding corrupts the data by expanding single bytes into multi-byte sequences (e.g., `\x80` becomes `\xc2\x80`)

## Relevant Context

Pickle data almost always contains bytes outside the ASCII range (0-127). The pickle protocol version 4 header starts with `\x80\x04`, where `\x80` is byte 128. When this is decoded using latin1 encoding, it becomes a string with character U+0080. When Python's default UTF-8 encoding is applied during `b.encode()`, this character is encoded as the two-byte sequence `\xc2\x80`, corrupting the pickle data.

The function is marked as deprecated and will be removed in Pydantic V3.0. However, while it exists, it should work correctly for users who haven't migrated yet. The workaround is to pass pickle data as bytes directly rather than as strings, or to use the modern Pydantic APIs.

Links:
- Pydantic Migration Guide: https://errors.pydantic.dev/2.10/migration/
- Source code: `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/deprecated/parse.py:53`

## Proposed Fix

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