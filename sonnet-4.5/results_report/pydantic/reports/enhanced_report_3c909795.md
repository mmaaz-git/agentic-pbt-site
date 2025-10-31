# Bug Report: pydantic.v1.types.ByteSize Silent Truncation of Fractional Byte Values

**Target**: `pydantic.v1.types.ByteSize`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ByteSize.validate() silently truncates fractional byte values to zero without warning, causing complete data loss for inputs like "0.5b" or "0.9b" which become 0 instead of raising an error or preserving the fractional value.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.v1.types import ByteSize


@given(scalar=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False))
def test_bytesize_precision_loss(scalar):
    bs = ByteSize.validate(f"{scalar}b")
    result = bs.to('b')
    assert result == scalar, f"ByteSize lost precision: {scalar}b -> {bs} -> {result}b"


if __name__ == "__main__":
    test_bytesize_precision_loss()
```

<details>

<summary>
**Failing input**: `scalar=0.5`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 13, in <module>
    test_bytesize_precision_loss()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 6, in test_bytesize_precision_loss
    def test_bytesize_precision_loss(scalar):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 9, in test_bytesize_precision_loss
    assert result == scalar, f"ByteSize lost precision: {scalar}b -> {bs} -> {result}b"
           ^^^^^^^^^^^^^^^^
AssertionError: ByteSize lost precision: 0.5b -> 0 -> 0.0b
Falsifying example: test_bytesize_precision_loss(
    scalar=0.5,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pydantic.v1.types import ByteSize

# Test fractional bytes
bs = ByteSize.validate("0.5b")
print(f"ByteSize.validate('0.5b') = {bs}")
print(f"bs.to('b') = {bs.to('b')}")
print()

# Test multiple fractional values
test_values = ["0.1b", "0.9b", "0.5b", "0.9999b"]
for value in test_values:
    bs = ByteSize.validate(value)
    result = bs.to('b')
    print(f"Input: {value:8} -> ByteSize: {bs} -> Back to bytes: {result}")
print()

# Test larger units with fractions
test_kb = ["1.7kb", "2.5mb", "0.5kb"]
for value in test_kb:
    bs = ByteSize.validate(value)
    unit = value[-2:] if value.endswith("mb") else value[-2:]
    result = bs.to(unit)
    print(f"Input: {value:8} -> ByteSize: {bs:7} -> Back to {unit}: {result}")
```

<details>

<summary>
Silent data loss: All fractional byte values become 0
</summary>
```
ByteSize.validate('0.5b') = 0
bs.to('b') = 0.0

Input: 0.1b     -> ByteSize: 0 -> Back to bytes: 0.0
Input: 0.9b     -> ByteSize: 0 -> Back to bytes: 0.0
Input: 0.5b     -> ByteSize: 0 -> Back to bytes: 0.0
Input: 0.9999b  -> ByteSize: 0 -> Back to bytes: 0.0

Input: 1.7kb    -> ByteSize:    1700 -> Back to kb: 1.7
Input: 2.5mb    -> ByteSize: 2500000 -> Back to mb: 2.5
Input: 0.5kb    -> ByteSize:     500 -> Back to kb: 0.5
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Silent Data Loss**: The regex pattern `r'^\s*(\d*\.?\d+)\s*(\w+)?'` at line 1089 explicitly accepts decimal numbers, indicating fractional inputs are intended to be valid. However, any fractional byte value less than 1 is silently truncated to 0 with no warning or error.

2. **Inconsistent Behavior**: The bug only affects byte-level fractions. Larger units work correctly (e.g., "0.5kb" becomes 500 bytes, "1.7kb" becomes 1700 bytes), but "0.5b" becomes 0 bytes instead of being rejected or preserved.

3. **Violates Round-Trip Property**: The fundamental property `ByteSize(x).to(unit) == x` fails for fractional bytes. Users expect data preservation or explicit rejection, not silent truncation.

4. **Documentation Ambiguity**: The Pydantic v1 documentation doesn't specify whether fractional bytes are supported, but the regex accepting them suggests they should be handled properly rather than silently discarded.

## Relevant Context

The root cause is in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/v1/types.py` at line 1117:
```python
return cls(int(float(scalar) * unit_mult))
```

For fractional bytes where `unit_mult=1`:
- `float("0.5") * 1 = 0.5`
- `int(0.5) = 0` (truncation occurs here)

The ByteSize class inherits from `int` (line 1092), which fundamentally cannot store fractional values. This architectural constraint means fractional bytes cannot be stored directly, but the current behavior of silently accepting and truncating them is problematic.

The BYTE_SIZES dictionary (lines 1073-1087) defines unit multipliers where 'b'=1, meaning only byte-level fractions are affected by this truncation issue.

Documentation link: https://docs.pydantic.dev/1.10/usage/types/#bytesize

## Proposed Fix

Since ByteSize inherits from int and cannot store fractional values, the solution is to explicitly reject fractional byte inputs that would lose precision:

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
+       raise errors.InvalidByteSize(f"ByteSize must be a whole number of bytes, got {scalar}{unit} = {result} bytes")
+   return cls(int(result))
```