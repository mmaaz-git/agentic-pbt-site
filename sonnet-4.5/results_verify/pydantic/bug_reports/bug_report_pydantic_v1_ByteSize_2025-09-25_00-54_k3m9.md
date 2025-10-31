# Bug Report: ByteSize Precision Loss on Fractional Input

**Target**: `pydantic.v1.types.ByteSize`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ByteSize.validate()` silently truncates fractional byte values to integers, causing precision loss. When parsing inputs like "0.5b" or "1.7kb", the fractional part is discarded without warning, leading to incorrect values when converting back.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.v1.types import ByteSize


@given(scalar=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False))
def test_bytesize_precision_loss(scalar):
    bs = ByteSize.validate(f"{scalar}b")
    result = bs.to('b')
    assert result == scalar, f"ByteSize lost precision: {scalar}b -> {bs} -> {result}b"
```

**Failing input**: `scalar=0.5`

## Reproducing the Bug

```python
from pydantic.v1.types import ByteSize

bs = ByteSize.validate("0.5b")
print(bs)
print(bs.to('b'))
```

**Output**:
```
0
0.0
```

**Expected**: Either reject fractional inputs with an error, or preserve the fractional value (though this conflicts with `ByteSize` inheriting from `int`).

## Why This Is A Bug

The `ByteSize` class accepts fractional input strings (e.g., "0.5b", "1.7kb") but silently truncates them to integers via `int(float(scalar) * unit_mult)` at line 1117. This violates the principle of least surprise:

1. Users providing "0.5kb" expect the value to be preserved
2. The round-trip property `ByteSize(x).to(unit) == x` is violated
3. No error or warning is raised about the precision loss

## Fix

The issue is in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/v1/types.py` at line 1117:

```diff
@classmethod
def validate(cls, v: StrIntFloat) -> 'ByteSize':
    try:
        return cls(int(v))
    except ValueError:
        pass

    str_match = byte_string_re.match(str(v))
    if str_match is None:
        raise errors.InvalidByteSize()

    scalar, unit = str_match.groups()
    if unit is None:
        unit = 'b'

    try:
        unit_mult = BYTE_SIZES[unit.lower()]
    except KeyError:
        raise errors.InvalidByteSizeUnit(unit=unit)

-   return cls(int(float(scalar) * unit_mult))
+   result = float(scalar) * unit_mult
+   if result != int(result):
+       raise errors.InvalidByteSize(f"ByteSize must be a whole number, got {result}")
+   return cls(int(result))
```

Alternatively, use `round()` instead of `int()` to reduce precision loss:

```diff
-   return cls(int(float(scalar) * unit_mult))
+   return cls(round(float(scalar) * unit_mult))
```