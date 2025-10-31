# Bug Report: Excluded Fields Break Round-Trip Invariant

**Target**: `pydantic.Field(exclude=True)`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Fields marked with `exclude=True` are excluded from `model_dump()` output but are still required (or reset to defaults) during `model_validate()`. This breaks the fundamental invariant that `model_validate(model_dump(m))` should equal `m` for all models.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel, Field

@given(
    public_value=st.integers(),
    private_value=st.integers()
)
def test_excluded_field_roundtrip(public_value, private_value):
    """
    PROPERTY: model_validate(model_dump(m)) should equal m, even with excluded fields.
    """
    class ExcludeModel(BaseModel):
        public: int
        private: int = Field(default=0, exclude=True)

    model = ExcludeModel(public=public_value, private=private_value)

    dumped = model.model_dump()
    restored = ExcludeModel.model_validate(dumped)

    assert model.private == restored.private
    assert model == restored
```

**Failing input**: Any values (e.g., `public_value=1, private_value=2`)

## Reproducing the Bug

```python
from pydantic import BaseModel, Field

class ExcludeModel(BaseModel):
    public: int
    private: int = Field(default=0, exclude=True)

model = ExcludeModel(public=1, private=2)
print(f"Original: public={model.public}, private={model.private}")

dumped = model.model_dump()
print(f"Dumped: {dumped}")

restored = ExcludeModel.model_validate(dumped)
print(f"Restored: public={restored.public}, private={restored.private}")
```

Output:
```
Original: public=1, private=2
Dumped: {'public': 1}
Restored: public=1, private=0
```

The excluded field's value (2) is lost and reset to the default (0).

If the excluded field has no default:
```python
class RequiredExcludeModel(BaseModel):
    public: int
    private: int = Field(exclude=True)

model = RequiredExcludeModel(public=1, private=2)
dumped = model.model_dump()

restored = RequiredExcludeModel.model_validate(dumped)
```

Output:
```
ValidationError: 1 validation error for RequiredExcludeModel
private
  Field required [type=missing, input_value={'public': 1}, input_type=dict]
```

## Why This Is A Bug

The `exclude=True` parameter is typically used for:
1. Private fields that shouldn't be serialized to external consumers
2. Sensitive data that should be kept internal
3. Computed or cached values that don't need serialization

However, these fields still have values that should be preserved during internal operations like:
- `model.model_copy()`
- Database round-trips
- Internal serialization/deserialization

The current behavior breaks the fundamental Pydantic invariant that all other field types satisfy. Users expect `model_validate(model_dump(m)) == m`, but excluded fields violate this.

## Fix

Option 1: Add `model_dump(mode='internal')` that includes excluded fields:
```python
dumped = model.model_dump(mode='internal')
restored = MyModel.model_validate(dumped)
```

Option 2: Make `exclude` only apply to external serialization (JSON, dict for API):
```diff
- dumped = model.model_dump()  # Excludes 'exclude=True' fields
+ dumped = model.model_dump()  # Includes all fields
+ json_dumped = model.model_dump_json()  # Excludes 'exclude=True' fields
```

Option 3: Document the limitation and provide workaround:
```python
dumped = model.model_dump(exclude=None)
restored = MyModel.model_validate(dumped)
```

The best solution is Option 1 or 2, which maintains the round-trip invariant for internal use while still providing exclusion for external serialization. The current behavior forces users to manually track excluded field values if they need round-trip capability.