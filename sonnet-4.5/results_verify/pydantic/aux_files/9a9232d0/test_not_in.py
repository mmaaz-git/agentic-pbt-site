from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform

# Simple test of the not_in constraint
forbidden = {1, 2, 3}

class TestModel(BaseModel):
    field: int = transform(lambda x: x).not_in(forbidden)

print("Testing not_in constraint with forbidden values {1, 2, 3}:")
for val in [1, 2, 3, 4, 5]:
    try:
        model = TestModel(field=val)
        print(f"Value {val}: ACCEPTED (BUG if in forbidden set)")
    except ValidationError as e:
        print(f"Value {val}: REJECTED (correct if in forbidden set)")