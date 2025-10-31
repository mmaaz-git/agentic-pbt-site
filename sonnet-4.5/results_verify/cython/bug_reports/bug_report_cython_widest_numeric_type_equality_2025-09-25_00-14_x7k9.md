# Bug Report: Cython widest_numeric_type Equality Comparison

**Target**: `Cython.Compiler.PyrexTypes.widest_numeric_type`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `widest_numeric_type` function violates the commutativity property: `widest_numeric_type(a, a)` called twice returns objects that don't compare equal using `==`, even though they should be the same object.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import Cython.Compiler.PyrexTypes as PT

NUMERIC_TYPES = [
    PT.c_uchar_type, PT.c_ushort_type, PT.c_uint_type, PT.c_ulong_type,
    PT.c_ulonglong_type, PT.c_char_type, PT.c_short_type, PT.c_int_type,
    PT.c_long_type, PT.c_longlong_type, PT.c_schar_type, PT.c_sshort_type,
    PT.c_sint_type, PT.c_slong_type, PT.c_slonglong_type,
    PT.c_float_type, PT.c_double_type, PT.c_longdouble_type,
    PT.c_float_complex_type, PT.c_double_complex_type, PT.c_longdouble_complex_type,
]

@given(st.sampled_from(NUMERIC_TYPES), st.sampled_from(NUMERIC_TYPES))
@settings(max_examples=1000)
def test_widest_numeric_type_commutative(type1, type2):
    result1 = PT.widest_numeric_type(type1, type2)
    result2 = PT.widest_numeric_type(type2, type1)
    assert result1 == result2
```

**Failing input**: `type1=<CNumericType short>, type2=<CNumericType short>`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/cython/site-packages')

import Cython.Compiler.PyrexTypes as PT

t = PT.c_short_type

result1 = PT.widest_numeric_type(t, t)
result2 = PT.widest_numeric_type(t, t)

print(f"result1 is result2: {result1 is result2}")
print(f"result1 == result2: {result1 == result2}")

assert result1 == result2
```

## Why This Is A Bug

The `widest_numeric_type` function is documented as returning "the narrowest type encompassing both" input types. When called with the same type twice, it should return a consistent result. The commutativity property is a fundamental mathematical property that should hold for type operations.

The bug occurs because:

1. When `widest_numeric_type(c_short_type, c_short_type)` is called, the function reaches line 5389 in PyrexTypes.py
2. The comparison at line 5360 (`if type1 == type2:`) fails even when both arguments are the same object
3. The function falls through to the final else clause (line 5392) and returns `type2`
4. Both calls return the same object (`c_short_type`), but the `==` comparison between the results fails

The root cause is in the `__eq__` implementation in `BaseType` (line 720):

```python
def __eq__(self, other):
    if isinstance(other, BaseType):
        return self.same_as_resolved_type(other)
    else:
        return False
```

And `same_as_resolved_type` (line 305):

```python
def same_as_resolved_type(self, other_type):
    return self == other_type or other_type is error_type
```

This creates a circular dependency: `__eq__` calls `same_as_resolved_type`, which calls `__eq__` again, leading to infinite recursion or incorrect comparison results.

## Fix

The issue is that `same_as_resolved_type` uses `==` which calls back to `__eq__`. The fix is to use identity comparison (`is`) as the base case:

```diff
--- a/Cython/Compiler/PyrexTypes.py
+++ b/Cython/Compiler/PyrexTypes.py
@@ -302,7 +302,7 @@ class BaseType:
         return self.same_as_resolved_type(other_type.resolve(), **kwds)

     def same_as_resolved_type(self, other_type):
-        return self == other_type or other_type is error_type
+        return self is other_type or other_type is error_type

     def subtype_of(self, other_type):
         return self.subtype_of_resolved_type(other_type.resolve())
```

This change makes the default implementation use identity comparison, which is correct for type objects that are typically singletons. Subclasses can still override `same_as_resolved_type` to provide more sophisticated comparison logic.