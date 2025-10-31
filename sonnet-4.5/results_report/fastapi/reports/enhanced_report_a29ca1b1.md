# Bug Report: FastAPI jsonable_encoder Crashes on Dictionaries with Tuple Keys

**Target**: `fastapi.encoders.jsonable_encoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `jsonable_encoder` function crashes with `TypeError: unhashable type: 'list'` when encoding dictionaries that have tuples as keys. The function internally converts tuples to lists during recursive encoding, then attempts to use these lists as dictionary keys, which fails because lists are unhashable.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.encoders import jsonable_encoder

@given(st.dictionaries(
    st.one_of(
        st.text(min_size=1, max_size=20),
        st.integers(),
        st.lists(st.integers(), min_size=1, max_size=3).map(tuple)
    ),
    st.integers(),
    max_size=5
))
def test_dict_with_various_key_types(obj):
    result = jsonable_encoder(obj)
    assert isinstance(result, dict)
    for key in result.keys():
        hash(key)

if __name__ == "__main__":
    test_dict_with_various_key_types()
```

<details>

<summary>
**Failing input**: `{(0,): 0}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 20, in <module>
    test_dict_with_various_key_types()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 5, in test_dict_with_various_key_types
    st.one_of(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 14, in test_dict_with_various_key_types
    result = jsonable_encoder(obj)
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/encoders.py", line 297, in jsonable_encoder
    encoded_dict[encoded_key] = encoded_value
    ~~~~~~~~~~~~^^^^^^^^^^^^^
TypeError: unhashable type: 'list'
Falsifying example: test_dict_with_various_key_types(
    obj={(0,): 0},
)
```
</details>

## Reproducing the Bug

```python
from fastapi.encoders import jsonable_encoder
import traceback

# Test case: dict with tuple key
obj = {(0,): 0}

try:
    result = jsonable_encoder(obj)
    print("Result:", result)
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
```

<details>

<summary>
TypeError: unhashable type: 'list' at line 297 of encoders.py
</summary>
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from fastapi.encoders import jsonable_encoder; import traceback; obj = {(0,): 0}; jsonable_encoder(obj)
                                                                                      ~~~~~~~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/encoders.py", line 297, in jsonable_encoder
    encoded_dict[encoded_key] = encoded_value
    ~~~~~~~~~~~~^^^^^^^^^^^^^
TypeError: unhashable type: 'list'
```
</details>

## Why This Is A Bug

This violates the documented contract that `jsonable_encoder` will "convert any object to something that can be encoded in JSON." Dictionaries with tuple keys are valid Python objects that should be handled gracefully rather than crashing with a confusing error message.

The root cause lies in the dictionary encoding logic (lines 264-298 of `encoders.py`):

1. When encoding a dictionary, the function recursively encodes both keys and values (lines 281-288)
2. During this recursive encoding, tuples are converted to lists (lines 299-315) because tuples need to become JSON arrays
3. The function then attempts to use these encoded keys (now lists) as dictionary keys at line 297: `encoded_dict[encoded_key] = encoded_value`
4. This fails because lists are unhashable and cannot be used as dictionary keys in Python

The error message "unhashable type: 'list'" is particularly confusing because users pass tuples (which are hashable), not lists. The lists are created internally by the encoder itself, making the error difficult to understand.

Since JSON only supports string keys in objects anyway, all non-string keys must eventually be converted to strings. The function already handles int, float, bool, and None keys successfully by keeping them as-is and letting `json.dumps()` convert them. However, it crashes on tuple keys due to the recursive encoding that transforms them into unhashable lists.

## Relevant Context

The JSON specification (RFC 7159) mandates that object keys must be strings. Python's standard `json.dumps()` handles non-string keys by either:
- Automatically converting basic types (int, float, bool, None) to strings
- Raising a TypeError for other types like tuples: "keys must be str, int, float, bool or None, not tuple"

FastAPI's `jsonable_encoder` aims to be more permissive, as stated in its documentation. It already successfully handles various non-string key types that `json.dumps()` would accept. The crash on tuple keys represents an inconsistency in this handling.

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/fastapi/encoders.py:297`
Documentation: https://fastapi.tiangolo.com/tutorial/encoder/

## Proposed Fix

```diff
--- a/fastapi/encoders.py
+++ b/fastapi/encoders.py
@@ -278,15 +278,9 @@ def jsonable_encoder(
                 and (value is not None or not exclude_none)
                 and key in allowed_keys
             ):
-                encoded_key = jsonable_encoder(
-                    key,
-                    by_alias=by_alias,
-                    exclude_unset=exclude_unset,
-                    exclude_none=exclude_none,
-                    custom_encoder=custom_encoder,
-                    sqlalchemy_safe=sqlalchemy_safe,
-                )
+                # JSON only supports string keys, convert non-string keys to strings
+                # to avoid issues with unhashable types after recursive encoding
+                encoded_key = key if isinstance(key, (str, int, float, bool, type(None))) else str(key)
                 encoded_value = jsonable_encoder(
                     value,
                     by_alias=by_alias,
```