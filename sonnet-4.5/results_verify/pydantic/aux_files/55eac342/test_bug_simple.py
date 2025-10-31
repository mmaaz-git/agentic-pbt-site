from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
from typing import Annotated


def test_gt_float_constraint():
    print("Testing gt(5.5) constraint on int field:")

    class Model(BaseModel):
        field: Annotated[int, validate_as(int).gt(5.5)]

    # Test values that should fail (≤ 5)
    print("\nValues that should fail validation (≤ 5):")
    for value in [-1000, 0, 5]:
        try:
            m = Model(field=value)
            print(f"  Value {value}: BUG - Did NOT raise ValidationError, got {m.field}")
        except ValidationError as e:
            print(f"  Value {value}: OK - Raised ValidationError as expected")

    # Test values that should pass (> 5.5, so integers ≥ 6)
    print("\nValues that should pass validation (> 5.5):")
    for value in [6, 10, 1000]:
        try:
            m = Model(field=value)
            print(f"  Value {value}: OK - Passed with value {m.field}")
        except ValidationError as e:
            print(f"  Value {value}: ERROR - Should have passed but raised: {e}")


if __name__ == "__main__":
    test_gt_float_constraint()