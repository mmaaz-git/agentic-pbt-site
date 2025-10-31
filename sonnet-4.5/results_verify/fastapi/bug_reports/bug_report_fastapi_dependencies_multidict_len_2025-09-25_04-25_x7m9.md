# Bug Report: FastAPI Dependencies _get_multidict_value TypeError

**Target**: `fastapi.dependencies.utils._get_multidict_value`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_get_multidict_value` function crashes with `TypeError` when processing sequence fields (e.g., `List[str]`) that receive non-sequence values from a regular `Mapping`, attempting to call `len()` on objects that don't support it.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from fastapi.dependencies.utils import _get_multidict_value
from fastapi._compat import create_model_field
from fastapi import params
from typing import List


@given(
    value=st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.dictionaries(st.text(), st.text()),
    )
)
@settings(max_examples=1000)
def test_get_multidict_value_no_crash_on_non_sequences(value):
    field = create_model_field(
        name="test_field",
        type_=List[str],
        default=[],
        alias="test_alias",
        required=False,
        field_info=params.Query()
    )

    values = {"test_alias": value}
    result = _get_multidict_value(field, values)
```

**Failing input**: Any non-sequence with no `len()` method, e.g. `42` or `{'key': 'value'}`

## Reproducing the Bug

```python
from fastapi.dependencies.utils import _get_multidict_value
from fastapi._compat import create_model_field
from fastapi import params
from typing import List

field = create_model_field(
    name="items",
    type_=List[str],
    default=[],
    alias="items",
    required=False,
    field_info=params.Query()
)

result = _get_multidict_value(field, {"items": 42})
```

**Output:**
```
TypeError: object of type 'int' has no len()
```

## Why This Is A Bug

The function signature accepts `values: Mapping[str, Any]`, which allows any value type. However, on line 731 of `utils.py`, the code attempts to call `len(value)` without first verifying that `value` is a sequence:

```python
or (is_sequence_field(field) and len(value) == 0)
```

This violates the robustness principle: the function should handle all inputs that match its type signature. While HTTP request params/headers typically contain strings, the function is also used with other data sources that could contain integers, floats, or other types.

The bug occurs when:
1. `field` is a sequence field (e.g., `List[str]`)
2. `values` is a regular `Mapping` (not `ImmutableMultiDict` or `Headers`)
3. The value for the field's alias is a non-None object without a `len()` method

## Fix

```diff
diff --git a/fastapi/dependencies/utils.py b/fastapi/dependencies/utils.py
index 1234567..abcdefg 100644
--- a/fastapi/dependencies/utils.py
+++ b/fastapi/dependencies/utils.py
@@ -728,7 +728,7 @@ def _get_multidict_value(
             and isinstance(value, str)  # For type checks
             and value == ""
         )
-        or (is_sequence_field(field) and len(value) == 0)
+        or (is_sequence_field(field) and hasattr(value, '__len__') and len(value) == 0)
     ):
         if field.required:
             return