# Bug Report: pandas.api.extensions.ExtensionDtype Hash-Equality Contract Violation

**Target**: `pandas.api.extensions.ExtensionDtype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

ExtensionDtype violates Python's fundamental hash-equality contract by allowing dtype instances to equal their string representations while having different hash values, breaking the requirement that equal objects must have equal hashes.

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


if __name__ == "__main__":
    test_extensiondtype_string_equality_implies_hash_equality()
```

<details>

<summary>
**Failing input**: `param=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 46, in <module>
    test_extensiondtype_string_equality_implies_hash_equality()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 37, in test_extensiondtype_string_equality_implies_hash_equality
    def test_extensiondtype_string_equality_implies_hash_equality(param):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 42, in test_extensiondtype_string_equality_implies_hash_equality
    assert hash(dtype) == hash(string_repr)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_extensiondtype_string_equality_implies_hash_equality(
    param=0,
)
```
</details>

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
print(f"hash(dtype): {hash(dtype)}")
print(f"hash(string_repr): {hash(string_repr)}")
print(f"hash(dtype) == hash(string_repr): {hash(dtype) == hash(string_repr)}")

# Dictionary lookup demonstration
d = {dtype: "value"}
print(f"\nd[dtype]: {d[dtype]}")
try:
    print(f"d[string_repr]: {d[string_repr]}")
except KeyError:
    print(f"d[string_repr]: KeyError - not found!")

# Show that despite being equal, they behave differently in sets
s = {dtype}
print(f"\nstring_repr in set containing dtype: {string_repr in s}")
print(f"But dtype == string_repr: {dtype == string_repr}")

# This violates Python's requirement that if a == b, then hash(a) == hash(b)
print(f"\nPython hash-equality contract violation:")
print(f"  dtype == string_repr: {dtype == string_repr}")
print(f"  hash(dtype) == hash(string_repr): {hash(dtype) == hash(string_repr)}")
print(f"  This violates the requirement that equal objects must have equal hashes!")
```

<details>

<summary>
Python hash-equality contract violation detected
</summary>
```
dtype == string_repr: True
hash(dtype): 4209543490
hash(string_repr): -2286461880653083368
hash(dtype) == hash(string_repr): False

d[dtype]: value
d[string_repr]: KeyError - not found!

string_repr in set containing dtype: False
But dtype == string_repr: True

Python hash-equality contract violation:
  dtype == string_repr: True
  hash(dtype) == hash(string_repr): False
  This violates the requirement that equal objects must have equal hashes!
```
</details>

## Why This Is A Bug

This violates Python's fundamental data model requirement that if two objects are equal (`a == b`), they must have the same hash value (`hash(a) == hash(b)`). The Python documentation explicitly states in the `__hash__` method specification: "The only required property is that objects which compare equal have the same hash value."

The bug occurs because:
1. `ExtensionDtype.__eq__()` at lines 133-137 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/base.py` returns True when comparing with a string matching the dtype's name by attempting to construct a dtype from the string
2. `ExtensionDtype.__hash__()` at line 147 only hashes the `_metadata` tuple attributes, not the name
3. The string is hashed using Python's built-in `str.__hash__()`, producing a completely different value

This causes practical problems:
- Dictionary lookups fail despite objects being equal: `dtype == string` but `d[dtype]` works while `d[string]` raises KeyError
- Sets incorrectly treat equal objects as different: `dtype == string` but `string not in {dtype}`
- Any code relying on Python's hash-equality contract may behave incorrectly

## Relevant Context

This bug affects all pandas built-in ExtensionDtype subclasses including:
- `CategoricalDtype`
- `DatetimeTZDtype`
- `PeriodDtype`
- `IntervalDtype`
- `SparseDtype`
- Any user-defined ExtensionDtype subclasses

The string comparison feature is documented in the `__eq__` method's docstring, stating that an ExtensionDtype instance is considered equal if "it's a string matching 'self.name'". However, this intentional feature creates an unintentional contract violation.

Documentation: [pandas ExtensionDtype API](https://pandas.pydata.org/docs/reference/api/pandas.api.extensions.ExtensionDtype.html)
Source code: [pandas/core/dtypes/base.py](https://github.com/pandas-dev/pandas/blob/main/pandas/core/dtypes/base.py)

## Proposed Fix

The cleanest fix is to remove string comparison from `__eq__` to maintain the hash-equality contract:

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