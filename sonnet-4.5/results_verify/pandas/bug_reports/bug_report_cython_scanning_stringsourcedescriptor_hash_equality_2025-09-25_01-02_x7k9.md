# Bug Report: Cython.Compiler.Scanning.StringSourceDescriptor Hash-Equality Contract Violation

**Target**: `Cython.Compiler.Scanning.StringSourceDescriptor`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`StringSourceDescriptor` violates Python's hash-equality contract: equal objects have different hashes, breaking set/dict operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from Cython.Compiler.Scanning import StringSourceDescriptor

@given(st.text(min_size=1), st.text(min_size=1))
def test_string_descriptor_same_name_same_hash(code1, code2):
    assume(code1 != code2)
    name = "test"
    desc1 = StringSourceDescriptor(name, code1)
    desc2 = StringSourceDescriptor(name, code2)
    if desc1 == desc2:
        assert hash(desc1) == hash(desc2)
```

**Failing input**: `code1='0', code2='O'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Scanning import StringSourceDescriptor

desc1 = StringSourceDescriptor("test", "0")
desc2 = StringSourceDescriptor("test", "O")

print("desc1 == desc2:", desc1 == desc2)
print("hash(desc1) == hash(desc2):", hash(desc1) == hash(desc2))

desc_set = {desc1}
print("desc2 in {desc1}:", desc2 in desc_set)

desc_dict = {desc1: "value1"}
desc_dict[desc2] = "value2"
print("len(desc_dict):", len(desc_dict))
```

**Output**:
```
desc1 == desc2: True
hash(desc1) == hash(desc2): False
desc2 in {desc1}: False  # Bug: should be True
len(desc_dict): 2        # Bug: should be 1
```

## Why This Is A Bug

Python's documentation states: "If two objects compare equal, they must have the same hash value."

Current implementation (Scanning.py:275-282):
- `__eq__`: Returns True if `self.name == other.name` (value-based)
- `__hash__`: Returns `id(self)` (identity-based)

This creates inconsistent behavior:
1. `desc1 == desc2` is True (names match)
2. `hash(desc1) != hash(desc2)` (different object identities)
3. Set/dict lookups fail: `desc2 in {desc1}` returns False despite equality

Compare with `FileSourceDescriptor` (Scanning.py:239-243):
- `__eq__`: Uses `self.filename == other.filename`
- `__hash__`: Uses `hash(self.filename)` ✅ Consistent!

## Fix

```diff
--- a/Cython/Compiler/Scanning.py
+++ b/Cython/Compiler/Scanning.py
@@ -273,9 +273,7 @@ class StringSourceDescriptor(SourceDescriptor):
         return "<stringsource>"

     def __hash__(self):
-        return id(self)
-        # Do not hash on the name, an identical string source should be the
-        # same object (name is often defaulted in other places)
-        # return hash(self.name)
+        return hash(self.name)

     def __eq__(self, other):
         return isinstance(other, StringSourceDescriptor) and self.name == other.name
```

This makes `__hash__` consistent with `__eq__` by hashing on `name`, matching the pattern used in `FileSourceDescriptor`.