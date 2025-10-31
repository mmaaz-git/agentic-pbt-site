# Bug Report: Cython.Compiler.TypeInference.find_spanning_type Commutativity Violation

**Target**: `Cython.Compiler.TypeInference.find_spanning_type`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `find_spanning_type(type1, type2)` function violates commutativity when one of the types is `error_type`. The result depends on argument order, leading to non-deterministic type inference and incorrect associativity.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import Cython.Compiler.TypeInference as TI
import Cython.Compiler.PyrexTypes as PT


all_type_objects = []
for name in dir(PT):
    if name.endswith('_type') and not name.startswith('_'):
        obj = getattr(PT, name)
        if hasattr(obj, '__class__') and not callable(obj) and not isinstance(obj, dict):
            if hasattr(obj, 'is_int') or hasattr(obj, 'is_error'):
                all_type_objects.append(obj)

type_strategy = st.sampled_from(all_type_objects)


@given(type_strategy, type_strategy)
@settings(max_examples=1000)
def test_find_spanning_type_commutativity(t1, t2):
    result_forward = TI.find_spanning_type(t1, t2)
    result_backward = TI.find_spanning_type(t2, t1)

    assert result_forward == result_backward
```

**Failing input**: `t1 = c_int_type, t2 = error_type`

## Reproducing the Bug

```python
import Cython.Compiler.TypeInference as TI
import Cython.Compiler.PyrexTypes as PT

t1 = PT.c_int_type
t2 = PT.error_type

result_12 = TI.find_spanning_type(t1, t2)
result_21 = TI.find_spanning_type(t2, t1)

print(f'find_spanning_type(int, error) = {result_12}')
print(f'find_spanning_type(error, int) = {result_21}')
print(f'Equal? {result_12 == result_21}')
```

**Output:**
```
find_spanning_type(int, error) = int
find_spanning_type(error, int) = <error>
Equal? False
```

## Why This Is A Bug

A spanning type operation finds a type that can represent both input types. This is fundamentally a symmetric operation - the order of arguments should not matter. The function's purpose (finding a type "assignable from both type1 and type2") is a symmetric definition.

**Root cause**: In `PyrexTypes._spanning_type()`, the assignability checks are asymmetric:

```python
elif type1.assignable_from(type2):
    return type1
elif type2.assignable_from(type1):
    return type2
```

When both types claim to be assignable from each other (as with `error_type`), whichever type appears first is returned, breaking commutativity.

**Impact**:
- Non-deterministic type inference depending on evaluation order
- Associativity violations: `spanning(spanning(a,b), c) != spanning(a, spanning(b,c))`
- Inconsistent compilation results when errors occur during type inference

## Fix

Add special handling for `error_type` in `PyrexTypes._spanning_type()`:

```diff
--- a/Cython/Compiler/PyrexTypes.py
+++ b/Cython/Compiler/PyrexTypes.py
@@ -XXXX,6 +XXXX,10 @@ def _spanning_type(type1, type2):
+    if type1.is_error:
+        return type1
+    if type2.is_error:
+        return type2
     if type1.is_numeric and type2.is_numeric:
         return widest_numeric_type(type1, type2)
```