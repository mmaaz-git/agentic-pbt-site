# Bug Report: pydantic.alias_generators Round-Trip Failure

**Target**: `pydantic.alias_generators.to_pascal` and `to_snake`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_pascal()` and `to_snake()` functions fail to round-trip correctly for snake_case strings containing single-letter segments, causing data loss when converting between naming conventions.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.alias_generators import to_pascal, to_snake

snake_case_strategy = st.from_regex(r'^[a-z]+(_[a-z]+)*$', fullmatch=True)

@given(snake_case_strategy)
def test_snake_to_pascal_to_snake_roundtrip(s):
    pascal = to_pascal(s)
    back_to_snake = to_snake(pascal)
    assert back_to_snake == s, \
        f"Round-trip failed: {s} -> {pascal} -> {back_to_snake}"
```

**Failing input**: `a_a`

## Reproducing the Bug

```python
from pydantic.alias_generators import to_pascal, to_snake

original = 'a_a'
pascal = to_pascal(original)
result = to_snake(pascal)

print(f"{original} -> {pascal} -> {result}")

assert result == original, f"Expected '{original}', got '{result}'"
```

Output:
```
a_a -> AA -> aa
AssertionError: Expected 'a_a', got 'aa'
```

Additional failing cases:
- `a_b_c` → `ABC` → `abc` (expected `a_b_c`)
- `x_y` → `XY` → `xy` (expected `x_y`)

## Why This Is A Bug

The functions are used as `alias_generator` in pydantic's `ConfigDict` (see `pydantic/config.py`), meaning users expect bidirectional conversion between snake_case field names and PascalCase aliases. The current implementation loses information when:

1. `to_pascal('a_a')` produces `'AA'` (consecutive uppercase letters)
2. `to_snake('AA')` produces `'aa'` instead of `'a_a'` because the regex patterns only insert underscores when an uppercase letter is followed by a lowercase letter

This violates the fundamental property that `to_snake(to_pascal(x)) == x` for valid snake_case inputs.

## Fix

The issue is in `to_snake()` at line 53. The regex `([A-Z]+)([A-Z][a-z])` handles sequences like `XMLParser` → `xml_parser`, but doesn't handle consecutive uppercase letters at the end like `AA`.

```diff
--- a/pydantic/alias_generators.py
+++ b/pydantic/alias_generators.py
@@ -50,6 +50,8 @@ def to_snake(camel: str) -> str:
         The converted string in snake_case.
     """
     # Handle the sequence of uppercase letters followed by a lowercase letter
     snake = re.sub(r'([A-Z]+)([A-Z][a-z])', lambda m: f'{m.group(1)}_{m.group(2)}', camel)
+    # Insert underscore between consecutive uppercase letters
+    snake = re.sub(r'([A-Z])([A-Z])', lambda m: f'{m.group(1)}_{m.group(2)}', snake)
     # Insert an underscore between a lowercase letter and an uppercase letter
     snake = re.sub(r'([a-z])([A-Z])', lambda m: f'{m.group(1)}_{m.group(2)}', snake)