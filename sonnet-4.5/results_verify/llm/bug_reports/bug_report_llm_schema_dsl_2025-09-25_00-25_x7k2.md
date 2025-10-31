# Bug Report: llm.utils.schema_dsl IndexError with Description-Only Fields

**Target**: `llm.utils.schema_dsl`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `schema_dsl` function crashes with `IndexError` when a field specification contains only a description (i.e., starts with `:`) without a field name.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import schema_dsl


@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
def test_schema_dsl_with_descriptions_only(descriptions):
    schema_input = ','.join(f':{desc}' for desc in descriptions)
    result = schema_dsl(schema_input)
    assert isinstance(result, dict)
```

**Failing input**: `:description` or `:some description` or any field starting with `:`

## Reproducing the Bug

```python
from llm.utils import schema_dsl

schema_input = ":description only"
result = schema_dsl(schema_input)
```

**Output:**
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "llm/utils.py", line 396, in schema_dsl
    field_name = field_parts[0].strip()
IndexError: list index out of range
```

## Why This Is A Bug

When a field in the DSL starts with `:`, the code splits it into `field_info` (empty string) and `description`. Then it tries to parse `field_info` by splitting on whitespace and accessing the first element:

```python
field_parts = field_info.strip().split()  # Returns [] when field_info is empty/whitespace
field_name = field_parts[0].strip()       # IndexError!
```

The function should either:
1. Skip fields that have no name
2. Raise a more informative `ValueError` explaining the invalid format
3. Handle empty field names gracefully

## Fix

```diff
    # Process field name and type
    field_parts = field_info.strip().split()
+
+   # Skip fields with no name
+   if not field_parts:
+       continue
+
    field_name = field_parts[0].strip()
```

Alternatively, raise a clear error:

```diff
    # Process field name and type
    field_parts = field_info.strip().split()
+
+   if not field_parts:
+       raise ValueError(f"Field specification has description but no field name: {repr(field)}")
+
    field_name = field_parts[0].strip()
```