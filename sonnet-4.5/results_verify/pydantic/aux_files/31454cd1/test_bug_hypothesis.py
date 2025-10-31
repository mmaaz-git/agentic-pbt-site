from hypothesis import given, strategies as st
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
from typing import Annotated
import pytest


@given(st.integers(min_value=-1000, max_value=5))
def test_gt_float_constraint_on_int(value):
    class Model(BaseModel):
        field: Annotated[int, validate_as(int).gt(5.5)]

    with pytest.raises(ValidationError):
        Model(field=value)


@given(st.integers(min_value=6, max_value=1000))
def test_gt_float_constraint_accepts_valid(value):
    class Model(BaseModel):
        field: Annotated[int, validate_as(int).gt(5.5)]

    m = Model(field=value)
    assert m.field == value


if __name__ == "__main__":
    # Run tests manually
    print("Testing values that should fail (≤ 5):")
    for value in [-1000, 0, 5]:
        try:
            test_gt_float_constraint_on_int(value)
            print(f"  Value {value}: PASSED (raised ValidationError as expected)")
        except AssertionError:
            print(f"  Value {value}: FAILED (did not raise ValidationError)")

    print("\nTesting values that should pass (> 5):")
    for value in [6, 10, 1000]:
        try:
            test_gt_float_constraint_accepts_valid(value)
            print(f"  Value {value}: PASSED")
        except Exception as e:
            print(f"  Value {value}: FAILED - {e}")