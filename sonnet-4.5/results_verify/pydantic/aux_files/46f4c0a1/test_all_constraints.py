from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated


class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(10)]

class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(10)]

class ModelLt(BaseModel):
    value: Annotated[int, validate_as(int).lt(10)]

class ModelLe(BaseModel):
    value: Annotated[int, validate_as(int).le(10)]


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


if __name__ == "__main__":
    models = {
        'Gt': ModelGt,
        'Ge': ModelGe,
        'Lt': ModelLt,
        'Le': ModelLe
    }

    print("Schema analysis for each constraint type:\n")
    for name, model_cls in models.items():
        schema = model_cls.__pydantic_core_schema__
        field_schema = schema['schema']['fields']['value']['schema']
        validator_count = count_validators_in_schema(schema)

        print(f"{name} constraint:")
        print(f"  Schema: {field_schema}")
        print(f"  Validator count: {validator_count}")
        print()

    # Test that the models work correctly
    print("Testing validation behavior:")
    print("ModelGt(value=11):", ModelGt(value=11).value)
    print("ModelGe(value=10):", ModelGe(value=10).value)
    print("ModelLt(value=9):", ModelLt(value=9).value)
    print("ModelLe(value=10):", ModelLe(value=10).value)