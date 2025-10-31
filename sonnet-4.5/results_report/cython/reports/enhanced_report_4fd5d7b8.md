# Bug Report: Cython.Compiler.Builtin.is_safe_compile_time_method Incorrectly Rejects Safe Methods for Types Without unsafe_compile_time_methods Entries

**Target**: `Cython.Compiler.Builtin.is_safe_compile_time_method`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function `is_safe_compile_time_method` incorrectly returns `False` for all methods on types that have entries in `inferred_method_return_types` but no entry in `unsafe_compile_time_methods` dictionary, preventing compile-time optimization of safe methods on bytearray, frozenset, dict, and memoryview types.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, example
from Cython.Compiler.Builtin import (
    is_safe_compile_time_method,
    inferred_method_return_types,
    unsafe_compile_time_methods
)

@given(st.sampled_from(['bytearray', 'frozenset', 'dict', 'memoryview']))
@example('bytearray')  # Ensure we test bytearray specifically
def test_types_with_inferred_methods_should_support_safe_methods(type_name):
    """Types with inferred methods but no unsafe_compile_time_methods entry
    should allow safe method evaluation."""

    methods = inferred_method_return_types.get(type_name, {})
    unsafe_methods = unsafe_compile_time_methods.get(type_name, set())

    # These types have inferred methods but no entry in unsafe_compile_time_methods
    assert type_name in inferred_method_return_types, f"{type_name} should be in inferred_method_return_types"
    assert type_name not in unsafe_compile_time_methods, f"{type_name} should not be in unsafe_compile_time_methods"

    # Check that all methods should be considered safe (since there's no unsafe list)
    for method_name in methods:
        result = is_safe_compile_time_method(type_name, method_name)
        assert result == True, f"Method {method_name} on {type_name} should be safe, but got {result}"

if __name__ == "__main__":
    test_types_with_inferred_methods_should_support_safe_methods()
```

<details>

<summary>
**Failing input**: `type_name='bytearray'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 30, in <module>
    test_types_with_inferred_methods_should_support_safe_methods()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 12, in test_types_with_inferred_methods_should_support_safe_methods
    @example('bytearray')  # Ensure we test bytearray specifically
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 27, in test_types_with_inferred_methods_should_support_safe_methods
    assert result == True, f"Method {method_name} on {type_name} should be safe, but got {result}"
           ^^^^^^^^^^^^^^
AssertionError: Method capitalize on bytearray should be safe, but got False
Falsifying explicit example: test_types_with_inferred_methods_should_support_safe_methods(
    type_name='bytearray',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Builtin import (
    is_safe_compile_time_method,
    inferred_method_return_types,
    unsafe_compile_time_methods
)

# Test bytearray
print("=== Testing bytearray ===")
print("bytearray in inferred_method_return_types:", 'bytearray' in inferred_method_return_types)
print("bytearray in unsafe_compile_time_methods:", 'bytearray' in unsafe_compile_time_methods)

methods = inferred_method_return_types['bytearray']
print(f"bytearray has {len(methods)} inferred methods (inherited from bytes)")

# Test a safe method: hex
print("\nTesting method 'hex' on bytearray:")
result = is_safe_compile_time_method('bytearray', 'hex')
print(f"is_safe_compile_time_method('bytearray', 'hex'): {result}")
print(f"Expected: True (hex is safe method inherited from bytes)")

# Test other affected types
print("\n=== Testing frozenset ===")
print("frozenset in inferred_method_return_types:", 'frozenset' in inferred_method_return_types)
print("frozenset in unsafe_compile_time_methods:", 'frozenset' in unsafe_compile_time_methods)
result = is_safe_compile_time_method('frozenset', 'copy')
print(f"is_safe_compile_time_method('frozenset', 'copy'): {result}")
print(f"Expected: True (copy is safe method)")

print("\n=== Testing dict ===")
print("dict in inferred_method_return_types:", 'dict' in inferred_method_return_types)
print("dict in unsafe_compile_time_methods:", 'dict' in unsafe_compile_time_methods)
result = is_safe_compile_time_method('dict', 'copy')
print(f"is_safe_compile_time_method('dict', 'copy'): {result}")
print(f"Expected: True (copy is safe method)")

print("\n=== Testing memoryview ===")
print("memoryview in inferred_method_return_types:", 'memoryview' in inferred_method_return_types)
print("memoryview in unsafe_compile_time_methods:", 'memoryview' in unsafe_compile_time_methods)
result = is_safe_compile_time_method('memoryview', 'hex')
print(f"is_safe_compile_time_method('memoryview', 'hex'): {result}")
print(f"Expected: True (hex is safe method)")

print("\n=== BUG CONFIRMED ===")
print("Function incorrectly returns False for safe methods on types that have")
print("inferred methods but no entry in unsafe_compile_time_methods dictionary.")
```

<details>

<summary>
Function incorrectly returns False for all methods on types with no unsafe_compile_time_methods entry
</summary>
```
=== Testing bytearray ===
bytearray in inferred_method_return_types: True
bytearray in unsafe_compile_time_methods: False
bytearray has 42 inferred methods (inherited from bytes)

Testing method 'hex' on bytearray:
is_safe_compile_time_method('bytearray', 'hex'): False
Expected: True (hex is safe method inherited from bytes)

=== Testing frozenset ===
frozenset in inferred_method_return_types: True
frozenset in unsafe_compile_time_methods: False
is_safe_compile_time_method('frozenset', 'copy'): False
Expected: True (copy is safe method)

=== Testing dict ===
dict in inferred_method_return_types: True
dict in unsafe_compile_time_methods: False
is_safe_compile_time_method('dict', 'copy'): False
Expected: True (copy is safe method)

=== Testing memoryview ===
memoryview in inferred_method_return_types: True
memoryview in unsafe_compile_time_methods: False
is_safe_compile_time_method('memoryview', 'hex'): False
Expected: True (hex is safe method)

=== BUG CONFIRMED ===
Function incorrectly returns False for safe methods on types that have
inferred methods but no entry in unsafe_compile_time_methods dictionary.
```
</details>

## Why This Is A Bug

The function `is_safe_compile_time_method` is designed to determine whether a method on a built-in type can be safely evaluated at compile time. According to the code comments in `unsafe_compile_time_methods` (lines 658-661), this dictionary only lists "unsafe and non-portable methods" for types that have literal representations and non-None return types.

The bug occurs because the function incorrectly interprets the absence of an entry in `unsafe_compile_time_methods` as meaning "not a literal type", when it actually means "this type has no unsafe methods". The faulty logic at lines 696-699:

```python
unsafe_methods = unsafe_compile_time_methods.get(builtin_type_name)
if unsafe_methods is None:
    # Not a literal type.
    return False
```

This causes the function to reject ALL methods for four types that ARE literal types (present in `inferred_method_return_types`) but have no unsafe methods listed:
- **bytearray**: Inherits 42 methods from bytes (line 632: `inferred_method_return_types['bytearray'].update(inferred_method_return_types['bytes'])`)
- **frozenset**: Inherits methods from set (line 633)
- **dict**: Has 3 methods defined (copy, fromkeys, popitem)
- **memoryview**: Has 3 methods defined (cast, hex, tobytes)

These types have compile-time evaluable methods with known return types, but the function incorrectly prevents their optimization by returning `False` for all their methods.

## Relevant Context

The data structures involved:
- `inferred_method_return_types`: Maps type names to their methods and return types
- `unsafe_compile_time_methods`: Maps type names to sets of unsafe methods
- Some types inherit methods through `.update()` calls (lines 632-633)

The comment at line 698 "Not a literal type" is misleading - the absence of an entry doesn't mean the type isn't literal, it means the type has no unsafe methods. Types like `tuple` have an empty set entry `'tuple': set()` (line 676) to indicate they are literal types with no unsafe methods, but bytearray, frozenset, dict, and memoryview lack even this empty entry.

Code references:
- Function definition: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/Builtin.py:695`
- Method inheritance: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/Builtin.py:632-633`

## Proposed Fix

```diff
--- a/Cython/Compiler/Builtin.py
+++ b/Cython/Compiler/Builtin.py
@@ -693,12 +693,14 @@ unsafe_compile_time_methods = {


 def is_safe_compile_time_method(builtin_type_name: str, method_name: str):
-    unsafe_methods = unsafe_compile_time_methods.get(builtin_type_name)
-    if unsafe_methods is None:
-        # Not a literal type.
+    # First check if this is a known type with inferred methods
+    known_methods = inferred_method_return_types.get(builtin_type_name)
+    if known_methods is None or method_name not in known_methods:
+        # Not a known method.
         return False
-    if method_name in unsafe_methods:
+    # Then check if this method is marked as unsafe
+    unsafe_methods = unsafe_compile_time_methods.get(builtin_type_name)
+    if unsafe_methods is not None and method_name in unsafe_methods:
         # Not a safe method.
         return False
-    known_methods = inferred_method_return_types.get(builtin_type_name)
-    if known_methods is None or method_name not in known_methods:
-        # Not a known method.
-        return False
     return True
```