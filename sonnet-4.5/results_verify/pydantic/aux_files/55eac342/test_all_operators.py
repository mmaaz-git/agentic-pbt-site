from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
from typing import Annotated

print("Testing all comparison operators with type mismatch (float constraint on int field):\n")

# Test gt(5.5)
class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5.5)]

try:
    m = ModelGt(value=5)
    print(f"gt(5.5) with value=5: BUG - passed validation, got {m.value}")
except ValidationError:
    print("gt(5.5) with value=5: OK - raised ValidationError")

# Test ge(5.5)
class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5.5)]

try:
    m = ModelGe(value=5)
    print(f"ge(5.5) with value=5: BUG - passed validation, got {m.value}")
except ValidationError:
    print("ge(5.5) with value=5: OK - raised ValidationError")

# Test lt(5.5)
class ModelLt(BaseModel):
    value: Annotated[int, validate_as(int).lt(5.5)]

try:
    m = ModelLt(value=6)
    print(f"lt(5.5) with value=6: BUG - passed validation, got {m.value}")
except ValidationError:
    print("lt(5.5) with value=6: OK - raised ValidationError")

# Test le(5.5)
class ModelLe(BaseModel):
    value: Annotated[int, validate_as(int).le(5.5)]

try:
    m = ModelLe(value=6)
    print(f"le(5.5) with value=6: BUG - passed validation, got {m.value}")
except ValidationError:
    print("le(5.5) with value=6: OK - raised ValidationError")

print("\nSummary: Only gt() fails to validate properly when there's a type mismatch.")