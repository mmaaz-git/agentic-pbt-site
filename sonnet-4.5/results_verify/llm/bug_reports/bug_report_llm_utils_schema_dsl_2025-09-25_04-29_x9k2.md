# Bug Report: llm.utils.schema_dsl Crashes on Empty Fields

**Target**: `llm.utils.schema_dsl`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `schema_dsl` function crashes with an `IndexError` when the input contains empty field specifications (e.g., consecutive commas or leading/trailing commas with whitespace).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import llm.utils as utils

@given(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po"))))
def test_schema_dsl_handles_malformed_input(schema_str):
    try:
        result = utils.schema_dsl(schema_str)
        assert isinstance(result, dict)
        assert "type" in result
        assert "properties" in result
    except (ValueError, IndexError) as e:
        assert isinstance(e, ValueError), f"Should raise ValueError, not {type(e).__name__}"
```

**Failing input**: `schema_dsl = "field1 str, , field2 int"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import schema_dsl

schema_str = "field1 str, , field2 int"

result = schema_dsl(schema_str)
```

**Output:**
```
IndexError: list index out of range
```

## Why This Is A Bug

The function splits on commas and processes each field, but doesn't validate that fields are non-empty before accessing `field_parts[0]`. When consecutive commas or whitespace-only fields appear in the input, the split results in empty strings that cause an IndexError.

User input for DSLs should be validated and provide clear error messages rather than crashing with IndexError.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -384,6 +384,10 @@ def schema_dsl(schema_dsl: str, multi: bool = False) -> Dict[str, Any]:

     # Process each field
     for field in fields:
+        # Skip empty fields
+        if not field.strip():
+            continue
+
         # Extract field name, type, and description
         if ":" in field:
             field_info, description = field.split(":", 1)
```

Alternatively, for stricter validation:

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -384,6 +384,10 @@ def schema_dsl(schema_dsl: str, multi: bool = False) -> Dict[str, Any]:

     # Process each field
     for field in fields:
+        # Validate field is non-empty
+        if not field.strip():
+            raise ValueError(f"Empty field in schema DSL: {repr(schema_dsl)}")
+
         # Extract field name, type, and description
         if ":" in field:
             field_info, description = field.split(":", 1)
```