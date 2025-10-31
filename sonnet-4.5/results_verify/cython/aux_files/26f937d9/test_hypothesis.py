#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import llm.utils as utils

@given(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po"))))
def test_schema_dsl_handles_malformed_input(schema_str):
    try:
        result = utils.schema_dsl(schema_str)
        assert isinstance(result, dict)
        assert "type" in result
        assert "properties" in result
    except (ValueError, IndexError) as e:
        assert isinstance(e, ValueError), f"Should raise ValueError, not {type(e).__name__}"

# Run the test
test_schema_dsl_handles_malformed_input()