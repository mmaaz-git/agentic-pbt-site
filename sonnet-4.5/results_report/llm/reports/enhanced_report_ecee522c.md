# Bug Report: llm.utils.schema_dsl IndexError on Empty Field Name Before Colon

**Target**: `llm.utils.schema_dsl`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `schema_dsl()` function crashes with an `IndexError` when given a field specification containing only whitespace before the colon separator, failing to validate input properly before attempting to parse field names.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

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

# Run the test
test_schema_dsl_empty_field_name()
```

<details>

<summary>
**Failing input**: `whitespace=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 11, in test_schema_dsl_empty_field_name
    result = schema_dsl(field_spec)
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py", line 396, in schema_dsl
    field_name = field_parts[0].strip()
                 ~~~~~~~~~~~^^^
IndexError: list index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 17, in <module>
    test_schema_dsl_empty_field_name()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 8, in test_schema_dsl_empty_field_name
    def test_schema_dsl_empty_field_name(whitespace):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 14, in test_schema_dsl_empty_field_name
    raise AssertionError("schema_dsl crashes on empty field name")
AssertionError: schema_dsl crashes on empty field name
Falsifying example: test_schema_dsl_empty_field_name(
    whitespace='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import schema_dsl

# Test case with empty field name before colon
result = schema_dsl(" : description")
print("Result:", result)
```

<details>

<summary>
IndexError: list index out of range at line 396
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/repo.py", line 7, in <module>
    result = schema_dsl(" : description")
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py", line 396, in schema_dsl
    field_name = field_parts[0].strip()
                 ~~~~~~~~~~~^^^
IndexError: list index out of range
```
</details>

## Why This Is A Bug

The `schema_dsl()` function is documented as "Build a JSON schema from a concise schema string" that accepts comma-separated or newline-separated field specifications. While the function expects field specifications in the format "field_name [type]: description", it fails to validate that a field name actually exists before attempting to access it. When given input like " : description" or ": description", the parsing logic creates an empty list when splitting the whitespace-only field info, then immediately attempts to access the first element of this empty list, causing an IndexError. A well-designed parsing function should validate its input and provide meaningful error messages for malformed specifications rather than crashing with a generic list index error.

## Relevant Context

The crash occurs in the field parsing logic at line 396 of `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py`. The function splits field specifications on colons to separate field info from descriptions (line 388), then attempts to extract the field name by splitting the field info on whitespace (line 395) and accessing the first element (line 396). When the field info contains only whitespace, `field_info.strip().split()` returns an empty list, causing the IndexError.

The function is part of the `llm` package's utility module and is used to create JSON schemas from simplified string representations. This makes proper input validation particularly important as users may provide various forms of input when constructing schemas programmatically.

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -393,6 +393,9 @@ def schema_dsl(schema_dsl: str, multi: bool = False) -> Dict[str, Any]:

         # Process field name and type
         field_parts = field_info.strip().split()
+        if not field_parts:
+            raise ValueError(f"Field specification has empty or missing field name: '{field}'")
+
         field_name = field_parts[0].strip()

         # Default type is string
```