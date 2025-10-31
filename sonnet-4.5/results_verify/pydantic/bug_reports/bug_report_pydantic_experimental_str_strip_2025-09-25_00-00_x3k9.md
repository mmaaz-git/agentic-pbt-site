# Bug Report: pydantic.experimental.pipeline str_strip Inconsistency

**Target**: `pydantic.experimental.pipeline._Pipeline.str_strip`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `str_strip()` method in pydantic's experimental pipeline API does not behave identically to Python's `str.strip()` method. It fails to strip certain whitespace characters that Python's `str.strip()` removes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from pydantic import BaseModel
from typing import Annotated
from pydantic.experimental.pipeline import validate_as


@given(st.text())
@settings(max_examples=1000)
@example('0\x1f')
@example('\x1f')
def test_str_strip_matches_python_str_strip(text):
    """Property: pipeline.str_strip() should behave identically to Python's str.strip()"""
    pipeline = validate_as(str).str_strip()

    class TestModel(BaseModel):
        field: Annotated[str, pipeline]

    model = TestModel(field=text)
    expected = text.strip()
    actual = model.field

    assert actual == expected, f"str_strip() mismatch: input={text!r}, expected={expected!r}, actual={actual!r}"
```

**Failing input**: `'0\x1f'` and `'\x1f'`

## Reproducing the Bug

```python
from pydantic import BaseModel
from typing import Annotated
from pydantic.experimental.pipeline import validate_as

pipeline = validate_as(str).str_strip()

class TestModel(BaseModel):
    field: Annotated[str, pipeline]

test_input = '0\x1f'
print(f"Input: {test_input!r}")
print(f"Python's str.strip(): {test_input.strip()!r}")

model = TestModel(field=test_input)
print(f"Pipeline str_strip(): {model.field!r}")
print(f"Match: {model.field == test_input.strip()}")
```

Output:
```
Input: '0\x1f'
Python's str.strip(): '0'
Pipeline str_strip(): '0\x1f'
Match: False
```

## Why This Is A Bug

The `str_strip()` method in `pipeline.py` line 310 is documented as transforming with `str.strip`:

```python
def str_strip(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
    return self.transform(str.strip)
```

This creates a contract: users expect `str_strip()` to behave exactly like Python's built-in `str.strip()` method.

However, the implementation in `_apply_transform()` (lines 427-431) replaces the `str.strip` function with pydantic_core's `strip_whitespace` schema attribute:

```python
if s['type'] == 'str':
    if func is str.strip:
        s = s.copy()
        s['strip_whitespace'] = True
        return s
```

The pydantic_core `strip_whitespace` attribute only strips ASCII whitespace characters (space, tab, newline, etc.), not all characters that Python's `str.strip()` removes. This violates the documented behavior and user expectations.

## Fix

Remove the optimization that replaces `str.strip` with `strip_whitespace`, and instead always use `str.strip` as a validator:

```diff
--- a/lib/python3.13/site-packages/pydantic/experimental/pipeline.py
+++ b/lib/python3.13/site-packages/pydantic/experimental/pipeline.py
@@ -426,11 +426,6 @@ def _apply_transform(
         return cs.no_info_plain_validator_function(func)

     if s['type'] == 'str':
-        if func is str.strip:
-            s = s.copy()
-            s['strip_whitespace'] = True
-            return s
-        elif func is str.lower:
+        if func is str.lower:
             s = s.copy()
             s['to_lower'] = True
             return s
```

Alternatively, if the optimization is desired for performance, document the limitation clearly and consider renaming the method to `str_strip_ascii()` or `str_strip_whitespace()` to accurately reflect its behavior.