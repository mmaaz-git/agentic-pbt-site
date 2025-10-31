#!/usr/bin/env python3
"""Examine schema structure in detail"""

from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
import pprint

class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5)]

class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5)]

class ModelLt(BaseModel):
    value: Annotated[int, validate_as(int).lt(5)]

class ModelLe(BaseModel):
    value: Annotated[int, validate_as(int).le(5)]

def examine_schema(schema, depth=0, name=""):
    indent = "  " * depth
    if isinstance(schema, dict):
        if schema.get('type') == 'function-after':
            print(f"{indent}function-after validator found at {name}")
            if 'schema' in schema:
                examine_schema(schema['schema'], depth+1, name + ".schema")
        elif schema.get('type') == 'model-fields':
            for field_name, field_info in schema.get('fields', {}).items():
                if 'schema' in field_info:
                    examine_schema(field_info['schema'], depth+1, f"{name}.fields.{field_name}")
        elif schema.get('type') in {'int', 'float', 'decimal'}:
            print(f"{indent}{schema['type']} schema at {name}")
            for constraint in ['gt', 'ge', 'lt', 'le']:
                if constraint in schema:
                    print(f"{indent}  Has {constraint} constraint: {schema[constraint]}")

print("Schema for Gt (value > 5):")
print("-" * 40)
examine_schema(ModelGt.__pydantic_core_schema__, name="ModelGt")

print("\nSchema for Ge (value >= 5):")
print("-" * 40)
examine_schema(ModelGe.__pydantic_core_schema__, name="ModelGe")

print("\nSchema for Lt (value < 5):")
print("-" * 40)
examine_schema(ModelLt.__pydantic_core_schema__, name="ModelLt")

print("\nSchema for Le (value <= 5):")
print("-" * 40)
examine_schema(ModelLe.__pydantic_core_schema__, name="ModelLe")

print("\n\nDetailed schema for Gt:")
print("="*50)
pprint.pprint(ModelGt.__pydantic_core_schema__)

print("\n\nDetailed schema for Ge:")
print("="*50)
pprint.pprint(ModelGe.__pydantic_core_schema__)