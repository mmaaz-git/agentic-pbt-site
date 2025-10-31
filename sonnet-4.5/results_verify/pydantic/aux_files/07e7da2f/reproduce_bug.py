import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from pydantic.experimental.pipeline import validate_as
from pydantic import BaseModel
from typing import Annotated

def analyze_schema(schema, indent=0):
    """Recursively analyze and print schema structure"""
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k in ('type', 'gt', 'ge', 'lt', 'le'):
                print(' ' * indent + f"{k}: {v}")
            elif k == 'schema':
                print(' ' * indent + "schema:")
                analyze_schema(v, indent + 2)
            elif isinstance(v, dict) and 'type' in v:
                print(' ' * indent + f"{k}:")
                analyze_schema(v, indent + 2)

class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5)]

class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5)]

class ModelLt(BaseModel):
    value: Annotated[int, validate_as(int).lt(10)]

class ModelLe(BaseModel):
    value: Annotated[int, validate_as(int).le(10)]

print("=== Gt constraint (CORRECT - uses else clause) ===")
schema_gt = ModelGt.__pydantic_core_schema__['schema']['fields']['value']['schema']
print("Full schema structure:")
print(schema_gt)
print("\nAnalysis:")
analyze_schema(schema_gt)

print("\n=== Ge constraint (BUG - missing else clause) ===")
schema_ge = ModelGe.__pydantic_core_schema__['schema']['fields']['value']['schema']
print("Full schema structure:")
print(schema_ge)
print("\nAnalysis:")
analyze_schema(schema_ge)

print("\n=== Lt constraint (BUG - missing else clause) ===")
schema_lt = ModelLt.__pydantic_core_schema__['schema']['fields']['value']['schema']
print("Full schema structure:")
print(schema_lt)
print("\nAnalysis:")
analyze_schema(schema_lt)

print("\n=== Le constraint (BUG - missing else clause) ===")
schema_le = ModelLe.__pydantic_core_schema__['schema']['fields']['value']['schema']
print("Full schema structure:")
print(schema_le)
print("\nAnalysis:")
analyze_schema(schema_le)

# Check for redundancy
def check_redundancy(schema, constraint_name):
    """Check if both schema constraint and validator are present"""
    has_schema_constraint = False
    has_validator = False

    def check(s):
        nonlocal has_schema_constraint, has_validator
        if isinstance(s, dict):
            if s.get('type') == 'int' and constraint_name in s:
                has_schema_constraint = True
            if 'validator' in str(s.get('type', '')):
                has_validator = True
            for v in s.values():
                if isinstance(v, (dict, list)):
                    check(v)
        elif isinstance(s, list):
            for item in s:
                check(item)

    check(schema)
    return has_schema_constraint and has_validator

print("\n=== Redundancy Check ===")
print(f"Gt has redundancy: {check_redundancy(schema_gt, 'gt')}")
print(f"Ge has redundancy: {check_redundancy(schema_ge, 'ge')}")
print(f"Lt has redundancy: {check_redundancy(schema_lt, 'lt')}")
print(f"Le has redundancy: {check_redundancy(schema_le, 'le')}")