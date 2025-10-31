"""
Theory: When transform is called with _FieldTypeMarker, the chain might be broken.
Let me trace through what happens:

1. transform(lambda x: x).not_in(forbidden) creates:
   - _ValidateAs(_FieldTypeMarker)  (from transform initialization)
   - _Transform(lambda x: x)        (from .transform(lambda x: x))
   - _Constraint(_NotIn(forbidden))  (from .not_in(forbidden))

2. In __get_pydantic_core_schema__, steps are processed:
   a. _ValidateAs(_FieldTypeMarker) -> returns handler(source_type) which is int
   b. _Transform(lambda x: x) -> wraps with no_info_plain_validator_function
   c. _Constraint(_NotIn(forbidden)) -> calls _apply_constraint

But wait, when _FieldTypeMarker is processed, if s is None (which it is initially),
it just returns handler(source_type) directly without chaining!

Let's check if that's the issue.
"""

from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform

# Test 1: Use not_in without transform - does it work?
print("Test 1: validate_as(int).not_in({1,2,3})")
from pydantic.experimental.pipeline import validate_as

class Model1(BaseModel):
    field: int = validate_as(int).not_in({1, 2, 3})

for val in [1, 2, 3, 4]:
    try:
        m = Model1(field=val)
        print(f"  Value {val}: ACCEPTED")
    except ValidationError:
        print(f"  Value {val}: REJECTED")

print("\nTest 2: transform(lambda x: x).validate_as(int).not_in({1,2,3})")

class Model2(BaseModel):
    field: int = transform(lambda x: x).validate_as(int).not_in({1, 2, 3})

for val in [1, 2, 3, 4]:
    try:
        m = Model2(field=val)
        print(f"  Value {val}: ACCEPTED")
    except ValidationError:
        print(f"  Value {val}: REJECTED")