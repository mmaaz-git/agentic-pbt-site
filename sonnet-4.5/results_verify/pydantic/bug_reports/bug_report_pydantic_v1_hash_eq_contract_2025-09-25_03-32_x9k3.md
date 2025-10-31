# Bug Report: pydantic.v1 BaseModel Hash/Equality Contract Violation

**Target**: `pydantic.v1.main.BaseModel.__eq__` and `generate_hash_function`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Pydantic.v1's `BaseModel.__eq__` compares models solely by their `dict()` representation, ignoring the model class. However, `generate_hash_function` includes the class in the hash. This violates Python's hash/equality contract: if `a == b`, then `hash(a)` must equal `hash(b)`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.v1 import BaseModel

def make_model_class(name: str):
    class DynamicModel(BaseModel):
        class Config:
            frozen = True
        x: int
    DynamicModel.__name__ = name
    DynamicModel.__qualname__ = name
    return DynamicModel

@given(st.integers(), st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=10))
def test_equal_objects_have_equal_hash(value, name1, name2):
    Model1 = make_model_class(name1)
    Model2 = make_model_class(name2)

    m1 = Model1(x=value)
    m2 = Model2(x=value)

    if m1 == m2:
        assert hash(m1) == hash(m2), \
            f"Equal objects must have equal hash: {m1} == {m2} but hash differs"
```

**Failing input**: Any case where two different model classes have the same fields and values.

## Reproducing the Bug

```python
from pydantic.v1 import BaseModel

class Model1(BaseModel):
    class Config:
        frozen = True
    x: int

class Model2(BaseModel):
    class Config:
        frozen = True
    x: int

m1 = Model1(x=42)
m2 = Model2(x=42)

assert m1 == m2

assert hash(m1) != hash(m2)

s = {m1}
assert m2 in s
```

The last assertion may fail depending on hash values, demonstrating broken set/dict behavior.

## Why This Is A Bug

**`__eq__` implementation (pydantic/v1/main.py:911-915):**
```python
def __eq__(self, other: Any) -> bool:
    if isinstance(other, BaseModel):
        return self.dict() == other.dict()
    else:
        return self.dict() == other
```

This ignores the class type, so `Model1(x=1) == Model2(x=1)` evaluates to True.

**`generate_hash_function` (pydantic/v1/main.py:102-106):**
```python
def hash_function(self_: Any) -> int:
    return hash(self_.__class__) + hash(tuple(self_.__dict__.values()))
```

This includes `hash(self_.__class__)`, so `hash(Model1(x=1))` typically differs from `hash(Model2(x=1))`.

**Python's hash/equality contract (from Python docs):**
> If two objects compare equal, they must have the same hash value.

This violation breaks:
1. **Set membership**: `{m1}` may or may not contain `m2` depending on hash collisions
2. **Dict keys**: `d[m1]` may not be retrievable via `d[m2]` even though `m1 == m2`
3. **Principle of least surprise**: Different types being equal is unexpected

## Fix

Include class comparison in `__eq__`:

```diff
--- a/pydantic/v1/main.py
+++ b/pydantic/v1/main.py
@@ -910,7 +910,7 @@ class BaseModel(Representation, metaclass=ModelMetaclass):

     def __eq__(self, other: Any) -> bool:
         if isinstance(other, BaseModel):
-            return self.dict() == other.dict()
+            return self.__class__ == other.__class__ and self.dict() == other.dict()
         else:
             return self.dict() == other
```

This ensures that only instances of the same model class can be equal, matching the hash function's behavior.