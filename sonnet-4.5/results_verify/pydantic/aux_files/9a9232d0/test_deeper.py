"""
Let me test whether the bug is actually in the fact that
ALL values pass (both in and out of the forbidden set)
"""

from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform, validate_as

forbidden = {1, 2, 3}

# Test 1: Does .in_() work correctly (opposite of not_in)?
print("Test 1: Using .in_() constraint:")
class ModelIn(BaseModel):
    field: int = validate_as(int).in_(forbidden)

for val in [1, 2, 3, 4, 5]:
    try:
        m = ModelIn(field=val)
        print(f"  Value {val}: ACCEPTED (should accept if in {forbidden})")
    except ValidationError as e:
        print(f"  Value {val}: REJECTED (should reject if not in {forbidden})")

print("\nTest 2: Using .not_in() constraint:")
class ModelNotIn(BaseModel):
    field: int = validate_as(int).not_in(forbidden)

for val in [1, 2, 3, 4, 5]:
    try:
        m = ModelNotIn(field=val)
        print(f"  Value {val}: ACCEPTED (should accept if not in {forbidden})")
    except ValidationError as e:
        print(f"  Value {val}: REJECTED (should reject if in {forbidden})")

print("\nTest 3: Using .eq() constraint:")
class ModelEq(BaseModel):
    field: int = validate_as(int).eq(2)

for val in [1, 2, 3]:
    try:
        m = ModelEq(field=val)
        print(f"  Value {val}: ACCEPTED (should only accept 2)")
    except ValidationError:
        print(f"  Value {val}: REJECTED (should reject if not 2)")

print("\nTest 4: Using .not_eq() constraint:")
class ModelNotEq(BaseModel):
    field: int = validate_as(int).not_eq(2)

for val in [1, 2, 3]:
    try:
        m = ModelNotEq(field=val)
        print(f"  Value {val}: ACCEPTED (should accept if not 2)")
    except ValidationError:
        print(f"  Value {val}: REJECTED (should reject if 2)")