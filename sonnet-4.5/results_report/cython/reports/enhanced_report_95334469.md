# Bug Report: Cython.Compiler.TypeInference.find_spanning_type Commutativity Violation with ErrorType

**Target**: `Cython.Compiler.TypeInference.find_spanning_type`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `find_spanning_type(type1, type2)` function violates commutativity when one argument is `error_type`, returning different results based on argument order. This causes non-deterministic type inference behavior during error recovery.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for Cython TypeInference commutativity with error_type"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
import Cython.Compiler.TypeInference as TI
import Cython.Compiler.PyrexTypes as PT

# Collect all type objects from PyrexTypes module
all_type_objects = []
for name in dir(PT):
    if name.endswith('_type') and not name.startswith('_'):
        obj = getattr(PT, name)
        if hasattr(obj, '__class__') and not callable(obj) and not isinstance(obj, dict):
            if hasattr(obj, 'is_int') or hasattr(obj, 'is_error'):
                all_type_objects.append(obj)

type_strategy = st.sampled_from(all_type_objects)

@given(type_strategy, type_strategy)
@example(PT.c_int_type, PT.error_type)  # Explicitly test the reported failing case
@settings(max_examples=1000)
def test_find_spanning_type_commutativity(t1, t2):
    """Test that find_spanning_type is commutative"""
    result_forward = TI.find_spanning_type(t1, t2)
    result_backward = TI.find_spanning_type(t2, t1)

    assert result_forward == result_backward, \
        f"Commutativity violated: find_spanning_type({t1}, {t2}) = {result_forward}, " \
        f"but find_spanning_type({t2}, {t1}) = {result_backward}"

# Run the test
if __name__ == '__main__':
    test_find_spanning_type_commutativity()
```

<details>

<summary>
**Failing input**: `t1=c_int_type, t2=error_type`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 36, in <module>
    test_find_spanning_type_commutativity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 23, in test_find_spanning_type_commutativity
    @example(PT.c_int_type, PT.error_type)  # Explicitly test the reported failing case
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 30, in test_find_spanning_type_commutativity
    assert result_forward == result_backward, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Commutativity violated: find_spanning_type(int, <error>) = int, but find_spanning_type(<error>, int) = <error>
Falsifying explicit example: test_find_spanning_type_commutativity(
    t1=<CNumericType int>,
    t2=<Cython.Compiler.PyrexTypes.ErrorType object at 0x70c2b49102f0>,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Cython TypeInference commutativity bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import Cython.Compiler.TypeInference as TI
import Cython.Compiler.PyrexTypes as PT

# Test commutativity with error_type and c_int_type
t1 = PT.c_int_type
t2 = PT.error_type

result_12 = TI.find_spanning_type(t1, t2)
result_21 = TI.find_spanning_type(t2, t1)

print(f'find_spanning_type(c_int_type, error_type) = {result_12}')
print(f'find_spanning_type(error_type, c_int_type) = {result_21}')
print(f'Are they equal? {result_12 == result_21}')

# Also test with other types to show this is systematic
print("\n--- Testing with other types ---")
test_types = [
    ('c_double_type', PT.c_double_type),
    ('c_float_type', PT.c_float_type),
    ('py_object_type', PT.py_object_type),
    ('c_void_type', PT.c_void_type),
]

for type_name, type_obj in test_types:
    r1 = TI.find_spanning_type(type_obj, PT.error_type)
    r2 = TI.find_spanning_type(PT.error_type, type_obj)
    print(f'{type_name}: ({r1}) vs ({r2}) - Equal? {r1 == r2}')
```

<details>

<summary>
Output demonstrates commutativity violation with error_type
</summary>
```
find_spanning_type(c_int_type, error_type) = int
find_spanning_type(error_type, c_int_type) = <error>
Are they equal? False

--- Testing with other types ---
c_double_type: (double) vs (<error>) - Equal? False
c_float_type: (double) vs (<error>) - Equal? False
py_object_type: (Python object) vs (Python object) - Equal? True
c_void_type: (void) vs (<error>) - Equal? False
```
</details>

## Why This Is A Bug

The `find_spanning_type` function is documented to "Return a type assignable from both type1 and type2" (PyrexTypes.py:5474), which is inherently a symmetric operation. The order of arguments should not affect the result since finding a common type that can represent both inputs is conceptually commutative.

The root cause lies in `ErrorType.same_as_resolved_type()` (PyrexTypes.py:4758) which unconditionally returns 1 (True), making `error_type` appear assignable from any type. This interacts badly with the asymmetric assignability checks in `_spanning_type()` (PyrexTypes.py:5502-5511):

```python
elif type1.assignable_from(type2):
    return type1
elif type2.assignable_from(type1):
    return type2
```

When `error_type` is the first argument, it claims to be assignable from anything and gets returned. When it's the second argument, the other type is checked first and may also claim assignability from `error_type`, causing that type to be returned instead.

While this only affects error recovery paths (compilation is already failing when `error_type` appears), it still violates expected mathematical properties and can lead to:
- Non-deterministic error messages depending on type evaluation order
- Inconsistent type inference during error recovery
- Potential associativity violations in complex type inference scenarios

## Relevant Context

The ErrorType class is documented as "Used to prevent propagation of error messages" (PyrexTypes.py:4740), and its promiscuous matching behavior is intentional for error recovery. However, this shouldn't break fundamental properties of type operations.

The issue only manifests with non-PyObject types. When paired with `py_object_type`, both orderings correctly return `py_object_type` due to special handling in `spanning_type()` (PyrexTypes.py:5480-5481).

Testing revealed that other pointer type combinations also violate commutativity (e.g., `char**` vs `void*`), suggesting broader issues with the assignability-based approach in `_spanning_type()`.

Relevant source locations:
- `find_spanning_type`: /home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/TypeInference.py:516-530
- `spanning_type`: /home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/PyrexTypes.py:5473-5488
- `_spanning_type`: /home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/PyrexTypes.py:5491-5520
- `ErrorType`: /home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/PyrexTypes.py:4739-4763

## Proposed Fix

Add special handling for `error_type` at the beginning of `_spanning_type()` to ensure it's always returned consistently regardless of argument order:

```diff
--- a/Cython/Compiler/PyrexTypes.py
+++ b/Cython/Compiler/PyrexTypes.py
@@ -5491,6 +5491,10 @@ def _spanning_type(type1, type2):
+    # Error type should always propagate consistently
+    if type1.is_error:
+        return type1
+    if type2.is_error:
+        return type2
     if type1.is_numeric and type2.is_numeric:
         return widest_numeric_type(type1, type2)
     elif type1.is_builtin_type:
```