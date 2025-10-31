# Bug Report: pandas.api.extensions.ExtensionDtype Hash-Equality Contract Violation

**Target**: `pandas.api.extensions.ExtensionDtype`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ExtensionDtype violates Python's hash-equality contract: when `dtype == string` is True (via string comparison), `hash(dtype) == hash(string)` is False. This breaks fundamental Python semantics and can cause subtle bugs when using dtypes in dictionaries or sets.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.api.extensions import ExtensionDtype


class ParametrizedDtype(ExtensionDtype):
    _metadata = ('param',)

    def __init__(self, param=0):
        self.param = param

    @property
    def name(self):
        return f"param[{self.param}]"

    @classmethod
    def construct_from_string(cls, string):
        if string.startswith("param[") and string.endswith("]"):
            param_str = string[6:-1]
            try:
                param = int(param_str)
                return cls(param)
            except ValueError:
                pass
        raise TypeError(f"Cannot construct from '{string}'")

    @classmethod
    def construct_array_type(cls):
        return np.ndarray

    @property
    def type(self):
        return object


@given(st.integers(-1000, 1000))
def test_extensiondtype_string_equality_implies_hash_equality(param):
    dtype = ParametrizedDtype(param)
    string_repr = dtype.name

    assert dtype == string_repr
    assert hash(dtype) == hash(string_repr)
```

**Failing input**: `param=0` (or any integer value)

## Reproducing the Bug

```python
import numpy as np
from pandas.api.extensions import ExtensionDtype


class ParametrizedDtype(ExtensionDtype):
    _metadata = ('param',)

    def __init__(self, param=0):
        self.param = param

    @property
    def name(self):
        return f"param[{self.param}]"

    @classmethod
    def construct_from_string(cls, string):
        if string.startswith("param[") and string.endswith("]"):
            param_str = string[6:-1]
            try:
                param = int(param_str)
                return cls(param)
            except ValueError:
                pass
        raise TypeError(f"Cannot construct from '{string}'")

    @classmethod
    def construct_array_type(cls):
        return np.ndarray

    @property
    def type(self):
        return object


dtype = ParametrizedDtype(0)
string_repr = dtype.name

print(f"dtype == string_repr: {dtype == string_repr}")
print(f"hash(dtype) == hash(string_repr): {hash(dtype) == hash(string_repr)}")

d = {dtype: "value"}
print(f"d[dtype]: {d[dtype]}")
print(f"d[string_repr]: {d.get(string_repr, 'KeyError - not found!')}")
```

Output:
```
dtype == string_repr: True
hash(dtype) == hash(string_repr): False
d[dtype]: value
d[string_repr]: KeyError - not found!
```

## Why This Is A Bug

Python's data model requires that if `a == b`, then `hash(a) == hash(b)`. This invariant is violated because:

1. `ExtensionDtype.__eq__()` returns True when comparing with a string matching the dtype's name (lines 133-137 in base.py)
2. `ExtensionDtype.__hash__()` only hashes the `_metadata` attributes (line 147 in base.py)
3. The string is hashed by Python's built-in `str.__hash__()` which produces a different value

This affects all built-in pandas dtypes (CategoricalDtype, PeriodDtype, IntervalDtype, etc.) and any user-defined ExtensionDtype subclasses.

This can cause bugs when:
- Using dtype instances and their string names interchangeably as dictionary keys
- Adding dtype instances and strings to the same set
- Any code that relies on the fundamental Python equality/hash contract

## Fix

The `__eq__` method should not return True for string comparisons, or the `__hash__` method needs to be updated to handle this case. The cleaner fix is to remove string comparison from `__eq__` and require explicit conversion:

```diff
--- a/pandas/core/dtypes/base.py
+++ b/pandas/core/dtypes/base.py
@@ -130,11 +130,6 @@ class ExtensionDtype:
         -------
         bool
         """
-        if isinstance(other, str):
-            try:
-                other = self.construct_from_string(other)
-            except TypeError:
-                return False
         if isinstance(other, type(self)):
             return all(
                 getattr(self, attr) == getattr(other, attr) for attr in self._metadata
```

Alternative fix (more complex, preserves string comparison):
```diff
--- a/pandas/core/dtypes/base.py
+++ b/pandas/core/dtypes/base.py
@@ -144,7 +144,18 @@ class ExtensionDtype:
     def __hash__(self) -> int:
         # for python>=3.10, different nan objects have different hashes
         # we need to avoid that and thus use hash function with old behavior
-        return object_hash(tuple(getattr(self, attr) for attr in self._metadata))
+        # Hash must match __eq__ behavior which allows string comparison
+        return hash(self.name)
```

The first fix is recommended as it's cleaner and avoids the ambiguity of dtype instances being equal to strings.