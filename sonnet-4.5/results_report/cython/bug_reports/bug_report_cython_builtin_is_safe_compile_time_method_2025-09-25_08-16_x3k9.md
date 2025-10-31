# Bug Report: Cython.Compiler.Builtin.is_safe_compile_time_method Missing Type Entries

**Target**: `Cython.Compiler.Builtin.is_safe_compile_time_method`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function `is_safe_compile_time_method` incorrectly returns `False` for types that have inferred method return types but no entry in the `unsafe_compile_time_methods` dictionary (bytearray, frozenset, dict, memoryview). This prevents compile-time evaluation of safe methods on these types.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler.Builtin import (
    is_safe_compile_time_method,
    inferred_method_return_types,
    unsafe_compile_time_methods
)

@given(st.sampled_from(['bytearray', 'frozenset', 'dict', 'memoryview']))
def test_types_with_inferred_methods_should_support_safe_methods(type_name):
    methods = inferred_method_return_types.get(type_name, {})
    unsafe_methods = unsafe_compile_time_methods.get(type_name, set())

    for method_name in methods:
        if method_name not in unsafe_methods:
            result = is_safe_compile_time_method(type_name, method_name)
            assert result == True, f"Method {method_name} on {type_name} should be safe"
```

**Failing input**: `type_name='bytearray'`, `method_name='hex'` (or any other safe method)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Builtin import (
    is_safe_compile_time_method,
    inferred_method_return_types,
    unsafe_compile_time_methods
)

print("bytearray in inferred_method_return_types:",
      'bytearray' in inferred_method_return_types)
print("bytearray in unsafe_compile_time_methods:",
      'bytearray' in unsafe_compile_time_methods)

methods = inferred_method_return_types['bytearray']
print(f"\nbytearray has {len(methods)} inferred methods (inherited from bytes)")

result = is_safe_compile_time_method('bytearray', 'hex')
print(f"\nis_safe_compile_time_method('bytearray', 'hex'): {result}")
print(f"Expected: True (hex is safe method inherited from bytes)")
print(f"Actual: {result}")

if not result:
    print("\n❌ BUG CONFIRMED: Function returns False for safe methods on types")
    print("   missing from unsafe_compile_time_methods dict")
```

Output:
```
bytearray in inferred_method_return_types: True
bytearray in unsafe_compile_time_methods: False
bytearray has 27 inferred methods (inherited from bytes)
is_safe_compile_time_method('bytearray', 'hex'): False
Expected: True (hex is safe method inherited from bytes)
Actual: False
❌ BUG CONFIRMED
```

## Why This Is A Bug

The function's logic at lines 695-699:

```python
def is_safe_compile_time_method(builtin_type_name: str, method_name: str):
    unsafe_methods = unsafe_compile_time_methods.get(builtin_type_name)
    if unsafe_methods is None:
        # Not a literal type.
        return False
```

This conflates two different concepts:
1. "Type is not a literal type"
2. "Type has no entry in the unsafe_methods dictionary"

However, bytearray, frozenset, dict, and memoryview ARE literal types (they're in `inferred_method_return_types`), but they have no entries in `unsafe_compile_time_methods`. The absence of an entry should mean "all methods are safe", not "type is not a literal type".

This causes the function to reject compile-time evaluation of safe methods like `bytearray.hex()`, `dict.copy()`, `frozenset.copy()`, etc.

## Fix

```diff
--- a/Cython/Compiler/Builtin.py
+++ b/Cython/Compiler/Builtin.py
@@ -693,12 +693,15 @@ unsafe_compile_time_methods = {


 def is_safe_compile_time_method(builtin_type_name: str, method_name: str):
+    known_methods = inferred_method_return_types.get(builtin_type_name)
+    if known_methods is None or method_name not in known_methods:
+        # Not a known method.
+        return False
     unsafe_methods = unsafe_compile_time_methods.get(builtin_type_name)
-    if unsafe_methods is None:
-        # Not a literal type.
-        return False
-    if method_name in unsafe_methods:
+    if unsafe_methods is not None and method_name in unsafe_methods:
         # Not a safe method.
         return False
-    known_methods = inferred_method_return_types.get(builtin_type_name)
-    if known_methods is None or method_name not in known_methods:
-        # Not a known method.
-        return False
     return True
```

The fix checks `inferred_method_return_types` first to determine if the type is a literal type with known methods, then checks `unsafe_compile_time_methods` to see if the specific method is unsafe. If the type has no unsafe_methods entry, all its inferred methods are considered safe.