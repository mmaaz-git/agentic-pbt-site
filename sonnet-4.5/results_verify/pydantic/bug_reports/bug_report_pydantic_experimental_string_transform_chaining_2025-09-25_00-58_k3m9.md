# Bug Report: Pydantic Experimental Pipeline String Transform Chaining

**Target**: `pydantic.experimental.pipeline._apply_transform`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When chaining multiple string transformations (str_lower, str_upper, str_strip) in a pipeline, only the first transformation is applied, not subsequent ones. The bug occurs because the code optimizes these transforms into schema-level flags, but pydantic-core only applies the first flag when multiple are present.

## Property-Based Test

```python
from typing import Annotated
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform


@given(st.text(min_size=1, max_size=100))
def test_transform_chaining_order(text):
    """Test that multiple transforms are applied in the correct order."""
    pipeline = transform(str.strip).transform(str.lower).transform(str.upper)

    class TestModel(BaseModel):
        field: Annotated[str, pipeline]

    model = TestModel(field=text)
    expected = text.strip().lower().upper()
    assert model.field == expected
```

**Failing input**: `'A'` (and most other inputs)

## Reproducing the Bug

```python
from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

pipeline = validate_as(str).str_lower().str_upper()

class TestModel(BaseModel):
    field: Annotated[str, pipeline]

test_input = "Hello"
model = TestModel(field=test_input)

expected = test_input.lower().upper()
actual = model.field

print(f"Input: '{test_input}'")
print(f"Expected: '{expected}'")
print(f"Actual: '{actual}'")
```

Output:
```
Input: 'Hello'
Expected: 'HELLO'
Actual: 'hello'
```

## Why This Is A Bug

The `_apply_transform` function in `pipeline.py` (lines 427-441) optimizes string transforms by setting schema-level flags:

```python
if s['type'] == 'str':
    if func is str.strip:
        s = s.copy()
        s['strip_whitespace'] = True
        return s
    elif func is str.lower:
        s = s.copy()
        s['to_lower'] = True
        return s
    elif func is str.upper:
        s = s.copy()
        s['to_upper'] = True
        return s
```

When you chain `.str_lower().str_upper()`, the resulting schema has both flags:
```python
{'type': 'str', 'to_lower': True, 'to_upper': True}
```

However, pydantic-core only applies the FIRST flag and ignores subsequent ones. This violates the expectation that transforms should compose sequentially.

The transforms work correctly when used alone, but fail when chained together.

## Fix

The optimization should only be used when there are no conflicting transform flags already set in the schema. If a transform flag already exists, fall through to use a validator function instead:

```diff
def _apply_transform(
    s: cs.CoreSchema | None, func: Callable[[Any], Any], handler: GetCoreSchemaHandler
) -> cs.CoreSchema:
    from pydantic_core import core_schema as cs

    if s is None:
        return cs.no_info_plain_validator_function(func)

    if s['type'] == 'str':
+       # Check if another string transform is already applied
+       has_transform = any(key in s for key in ['strip_whitespace', 'to_lower', 'to_upper'])
+
        if func is str.strip:
-           s = s.copy()
-           s['strip_whitespace'] = True
-           return s
+           if not has_transform:
+               s = s.copy()
+               s['strip_whitespace'] = True
+               return s
        elif func is str.lower:
-           s = s.copy()
-           s['to_lower'] = True
-           return s
+           if not has_transform:
+               s = s.copy()
+               s['to_lower'] = True
+               return s
        elif func is str.upper:
-           s = s.copy()
-           s['to_upper'] = True
-           return s
+           if not has_transform:
+               s = s.copy()
+               s['to_upper'] = True
+               return s

    return cs.no_info_after_validator_function(func, s)
```

This ensures that when transforms are chained, the subsequent transforms are applied as validator functions rather than conflicting schema flags.