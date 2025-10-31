import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.utils import schema_dsl

@given(st.text(max_size=20).filter(lambda s: s.strip() == ""))
def test_schema_dsl_empty_field_name(whitespace):
    field_spec = whitespace + ": some description"
    try:
        result = schema_dsl(field_spec)
        assert isinstance(result, dict)
    except IndexError:
        raise AssertionError("schema_dsl crashes on empty field name")

# Run the test
test_schema_dsl_empty_field_name()