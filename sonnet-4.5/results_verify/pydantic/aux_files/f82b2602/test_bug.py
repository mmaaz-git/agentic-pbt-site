from hypothesis import given, strategies as st, settings
from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types


@given(st.integers(min_value=-1000, max_value=1000))
@settings(max_examples=200)
def test_ge_constraint_schema_structure_matches_gt(value):
    int_schema = cs.int_schema()

    ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(value))
    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))

    assert ge_result['type'] == gt_result['type'], \
        f"Ge and Gt should produce same schema type, got {ge_result['type']} vs {gt_result['type']}"

# Run the test
test_ge_constraint_schema_structure_matches_gt()