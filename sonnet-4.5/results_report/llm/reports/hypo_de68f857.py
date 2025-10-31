import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import SharedOptions
from pydantic import ValidationError

@given(st.integers(min_value=101))
def test_logit_bias_out_of_range_error_message(out_of_range_value):
    """Property: out-of-range values should give specific error message"""
    try:
        SharedOptions(logit_bias={"123": out_of_range_value})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        error_msg = str(e)
        # Should see the specific range error, not generic message
        assert "between -100 and 100" in error_msg, f"Expected 'between -100 and 100' in error message, but got: {error_msg}"

# Run the test
if __name__ == "__main__":
    test_logit_bias_out_of_range_error_message()