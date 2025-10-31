from hypothesis import given, strategies as st, example
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

@given(st.integers(min_value=0, max_value=100))
@example(5)  # Specific example from the bug report
def test_ge_constraint_redundancy_hypothesis(threshold):
    """Property-based test showing that Ge/Lt/Le constraints work correctly but inefficiently.

    The bug doesn't cause incorrect behavior - it just applies validation twice:
    1. Once natively via the schema's 'ge' field (correct)
    2. Once via _check_func wrapper (redundant)
    """

    # Test with Ge constraint
    class ModelGe(BaseModel):
        field: Annotated[int, transform(lambda x: x).ge(threshold)]

    # Test valid values
    test_value = threshold + 10
    model = ModelGe(field=test_value)
    assert model.field == test_value

    # Test boundary value
    model_boundary = ModelGe(field=threshold)
    assert model_boundary.field == threshold

    # Test invalid value (should raise validation error)
    try:
        ModelGe(field=threshold - 1)
        assert False, "Should have raised validation error"
    except ValueError:
        pass  # Expected

    # Similar tests for Lt and Le constraints
    class ModelLt(BaseModel):
        field: Annotated[int, transform(lambda x: x).lt(threshold)]

    class ModelLe(BaseModel):
        field: Annotated[int, transform(lambda x: x).le(threshold)]

    # Lt tests
    if threshold > 0:
        model_lt = ModelLt(field=threshold - 1)
        assert model_lt.field == threshold - 1
        try:
            ModelLt(field=threshold)
            assert False, "Should have raised validation error for Lt"
        except ValueError:
            pass

    # Le tests
    model_le = ModelLe(field=threshold)
    assert model_le.field == threshold
    if threshold > 0:
        model_le2 = ModelLe(field=threshold - 1)
        assert model_le2.field == threshold - 1
    try:
        ModelLe(field=threshold + 1)
        assert False, "Should have raised validation error for Le"
    except ValueError:
        pass

# Run the test
if __name__ == "__main__":
    print("Running property-based test with Hypothesis...")
    print("Testing that Ge/Lt/Le constraints work correctly (despite redundant validation)")
    print("=" * 60)

    test_ge_constraint_redundancy_hypothesis()

    print("\n✓ All tests passed!")
    print("\nNote: The bug is about REDUNDANT validation, not incorrect behavior.")
    print("The constraints work correctly but apply validation twice internally.")