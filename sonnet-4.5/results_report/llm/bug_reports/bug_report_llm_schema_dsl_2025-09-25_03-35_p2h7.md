# Bug Report: llm.utils.schema_dsl IndexError on Empty Field Name

**Target**: `llm.utils.schema_dsl`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `schema_dsl()` function crashes with an `IndexError` when given a field specification with only a description (empty or whitespace-only field name before the colon).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import schema_dsl

@given(st.text(max_size=20).filter(lambda s: s.strip() == ""))
def test_schema_dsl_empty_field_name(whitespace):
    field_spec = whitespace + ": some description"
    try:
        result = schema_dsl(field_spec)
        assert isinstance(result, dict)
    except IndexError:
        raise AssertionError("schema_dsl crashes on empty field name")
```

**Failing input**: Any field specification where the part before the colon is empty or only whitespace, such as `" : description"` or `": description"`.

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import schema_dsl

result = schema_dsl(" : description")
```

This crashes with:
```
IndexError: list index out of range
```

## Why This Is A Bug

The function should either:
1. Gracefully handle invalid input with a clear error message
2. Skip fields with empty names
3. Raise a `ValueError` with a descriptive message

Instead, it crashes with an `IndexError` at line 396:
```python
field_name = field_parts[0].strip()
```

This happens because when a field is " : description":
1. Line 388 splits it into `field_info = " "` and `description = " description"`
2. Line 395 splits field_info: `field_parts = field_info.strip().split()` → `[]` (empty list)
3. Line 396 tries to access `field_parts[0]` → IndexError

While empty field names are arguably invalid input, crashing with an IndexError is poor error handling for a parsing function. Users should get a clear error message about what's wrong with their input.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -394,6 +394,9 @@ def schema_dsl(schema_dsl: str, multi: bool = False) -> Dict[str, Any]:
         # Process field name and type
         field_parts = field_info.strip().split()
+        if not field_parts:
+            raise ValueError(f"Field specification has empty field name: '{field}'")
+
         field_name = field_parts[0].strip()

         # Default type is string
```

This provides a clear error message when the field name is missing, making it easier for users to debug their schema DSL strings.