from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated
from hypothesis import given, strategies as st


class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(10)]


class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(10)]


def count_validators_in_schema(schema, path=""):
    count = 0
    if isinstance(schema, dict):
        if schema.get('type') == 'function-after' or schema.get('type') == 'no-info':
            count += 1
        for key, value in schema.items():
            count += count_validators_in_schema(value, f"{path}.{key}")
    elif isinstance(schema, (list, tuple)):
        for i, item in enumerate(schema):
            count += count_validators_in_schema(item, f"{path}[{i}]")
    return count


@given(st.integers(min_value=11))
def test_gt_ge_schema_consistency(x):
    gt_schema = ModelGt.__pydantic_core_schema__
    ge_schema = ModelGe.__pydantic_core_schema__

    gt_validators = count_validators_in_schema(gt_schema)
    ge_validators = count_validators_in_schema(ge_schema)

    assert gt_validators == ge_validators, \
        f"Gt has {gt_validators} validators but Ge has {ge_validators} validators"


if __name__ == "__main__":
    # Run the hypothesis test
    print("Running hypothesis test...")
    try:
        test_gt_ge_schema_consistency()
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Run the direct reproduction
    print("\nDirect reproduction:")
    gt_schema = ModelGt.__pydantic_core_schema__
    ge_schema = ModelGe.__pydantic_core_schema__

    print("Gt schema:", gt_schema['schema']['fields']['value']['schema'])
    print("Ge schema:", ge_schema['schema']['fields']['value']['schema'])

    print("\nValidation count comparison:")
    gt_count = count_validators_in_schema(gt_schema)
    ge_count = count_validators_in_schema(ge_schema)
    print(f"Gt validators: {gt_count}")
    print(f"Ge validators: {ge_count}")