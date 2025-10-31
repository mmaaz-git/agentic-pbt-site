# Bug Report: pydantic.alias_generators Number Handling

**Target**: `pydantic.alias_generators.to_snake`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_snake()` function incorrectly inserts underscores before digits when converting from PascalCase/camelCase, even when the original snake_case input did not have underscores before numbers.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.alias_generators import to_pascal, to_snake

snake_case_strategy = st.from_regex(r'^[a-z]+(_[a-z]+)*$', fullmatch=True)

@given(snake_case_strategy)
def test_snake_with_numbers_roundtrip(s):
    s_with_num = s + '1' if s else 'a1'

    pascal = to_pascal(s_with_num)
    back = to_snake(pascal)

    assert back == s_with_num, \
        f"Round-trip with number failed: {s_with_num} -> {pascal} -> {back}"
```

**Failing input**: `aa1`

## Reproducing the Bug

```python
from pydantic.alias_generators import to_pascal, to_snake

original = 'aa1'
pascal = to_pascal(original)
result = to_snake(pascal)

print(f"{original} -> {pascal} -> {result}")

assert result == original, f"Expected '{original}', got '{result}'"
```

Output:
```
aa1 -> Aa1 -> aa_1
AssertionError: Expected 'aa1', got 'aa_1'
```

Additional failing cases:
- `field1` → `Field1` → `field_1` (expected `field1`)
- `test2` → `Test2` → `test_2` (expected `test2`)
- `var3x` → `Var3X` → `var_3_x` (expected `var3x`)

## Why This Is A Bug

Field names with numbers are common in programming (e.g., `field1`, `test2`, `hash256`). The `to_snake()` function at line 59 inserts underscores between lowercase letters and digits:

```python
snake = re.sub(r'([a-z])([0-9])', lambda m: f'{m.group(1)}_{m.group(2)}', snake)
```

This transformation is not reversible - when converting `field1` → `Field1` → `field_1`, the original format is lost. This violates the round-trip property that users would expect when using these functions as alias generators.

## Fix

The issue is that `to_snake()` assumes all numbers should be separated by underscores, but this is not how snake_case typically works. Numbers are usually kept adjacent to letters unless there's a semantic boundary.

A potential fix would be to only insert underscores when converting from PascalCase where the number is preceded by an uppercase letter that isn't at a word boundary:

```diff
--- a/pydantic/alias_generators.py
+++ b/pydantic/alias_generators.py
@@ -56,8 +56,6 @@ def to_snake(camel: str) -> str:
     # Insert an underscore between a digit and an uppercase letter
     snake = re.sub(r'([0-9])([A-Z])', lambda m: f'{m.group(1)}_{m.group(2)}', snake)
-    # Insert an underscore between a lowercase letter and a digit
-    snake = re.sub(r'([a-z])([0-9])', lambda m: f'{m.group(1)}_{m.group(2)}', snake)
     # Replace hyphens with underscores to handle kebab-case
     snake = snake.replace('-', '_')
     return snake.lower()
```

However, this may break other use cases. A better approach might be to track the original format or document the limitation.