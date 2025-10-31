# Bug Report: fastapi.dependencies Path Parameter Default Value Inconsistency

**Target**: `fastapi.dependencies.utils.analyze_param`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `analyze_param` function exhibits inconsistent behavior when handling path parameters with default values. When using plain type annotations, it silently allows default values for path parameters. However, when using `Annotated` with `Path()`, it raises an `AssertionError`. This violates the documented constraint that "Path parameters cannot have default values" and breaks the principle of least surprise.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from typing import Annotated
import inspect
from fastapi.dependencies.utils import analyze_param
from fastapi import Path

@given(st.integers())
def test_path_param_default_consistency(default_value):
    # Case 1: Plain annotation
    details1 = analyze_param(
        param_name="item_id",
        annotation=int,
        value=default_value,
        is_path_param=True
    )

    # Case 2: Annotated with Path()
    path_info = Path()
    path_info.annotation = int

    details2 = analyze_param(
        param_name="item_id",
        annotation=Annotated[int, path_info],
        value=default_value,
        is_path_param=True
    )

    # Property: Both cases should behave the same
    assert type(details1.field.field_info) == type(details2.field.field_info)
```

**Failing input**: Any non-empty default value (e.g., `0`, `42`, `-1`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

import inspect
from typing import Annotated
from fastapi.dependencies.utils import analyze_param
from fastapi import Path

details1 = analyze_param(
    param_name="item_id",
    annotation=int,
    value=42,
    is_path_param=True
)
print(f"Case 1 (plain): Success - Field created: {details1.field is not None}")

path_info = Path()
path_info.annotation = int

try:
    details2 = analyze_param(
        param_name="item_id",
        annotation=Annotated[int, path_info],
        value=42,
        is_path_param=True
    )
    print(f"Case 2 (Annotated): Success")
except AssertionError as e:
    print(f"Case 2 (Annotated): AssertionError: {e}")
```

Output:
```
Case 1 (plain): Success - Field created: True
Case 2 (Annotated): AssertionError: Path parameters cannot have default values
```

## Why This Is A Bug

This is a contract violation because:

1. **Inconsistent behavior**: The same logical operation (creating a path parameter with a default value) produces different results depending on whether the user uses plain annotations or `Annotated`.

2. **Documented constraint violation**: The assertion message states "Path parameters cannot have default values", but Case 1 violates this constraint without raising an error.

3. **Silent failure**: Case 1 doesn't raise an error but also doesn't actually use the default value (field.default becomes `PydanticUndefined`), leading to surprising behavior.

The code at `fastapi/dependencies/utils.py:394-396` only checks for default values when `field_info` is already set (from `Annotated`), but skips this check when `field_info` is created later at line 448-452.

## Fix

```diff
--- a/fastapi/dependencies/utils.py
+++ b/fastapi/dependencies/utils.py
@@ -442,6 +442,11 @@ def analyze_param(
     # Handle default assignations, neither field_info nor depends was not found in Annotated nor default value
     elif field_info is None and depends is None:
         default_value = value if value is not inspect.Signature.empty else RequiredParam
+        # Validate path parameters cannot have defaults BEFORE creating field_info
+        if is_path_param and value is not inspect.Signature.empty:
+            raise ValueError(
+                f"Path parameter {param_name!r} cannot have a default value"
+            )
         if is_path_param:
             # We might check here that `default_value is RequiredParam`, but the fact is that the same
             # parameter might sometimes be a path parameter and sometimes not. See
```