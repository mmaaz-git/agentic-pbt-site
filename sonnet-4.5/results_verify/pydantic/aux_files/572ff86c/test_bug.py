import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from hypothesis import given, strategies as st
from pydantic.experimental.pipeline import validate_as
from pydantic import BaseModel
from typing import Annotated

@given(st.integers())
def test_ge_constraint_not_redundant(x):
    """Property: Ge constraint should only validate once"""

    class ModelGe(BaseModel):
        value: Annotated[int, validate_as(int).ge(5)]

    schema = ModelGe.__pydantic_core_schema__

    has_schema_constraint = False
    has_validator_function = False

    def check_schema(s):
        nonlocal has_schema_constraint, has_validator_function
        if isinstance(s, dict):
            if s.get('type') == 'int' and 'ge' in s:
                has_schema_constraint = True
            if s.get('type') in ('no-info-after-validator-function', 'no-info-plain-validator-function'):
                has_validator_function = True
            for v in s.values():
                if isinstance(v, dict):
                    check_schema(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            check_schema(item)

    check_schema(schema)

    assert not (has_schema_constraint and has_validator_function), \
        "Both schema constraint and validator function present - redundant validation!"

# Run the test
test_ge_constraint_not_redundant()
print("Test completed")