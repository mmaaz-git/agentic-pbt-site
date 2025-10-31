#!/usr/bin/env python3
"""Property-based test for validate_logit_bias bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from llm.default_plugins.openai_models import SharedOptions
import pytest

@given(
    st.dictionaries(
        st.integers(min_value=0, max_value=100000).map(str),
        st.integers().filter(lambda x: x < -100 or x > 100),
        min_size=1
    )
)
@settings(max_examples=10)  # Limit examples for brevity
def test_validate_logit_bias_error_messages(logit_bias_dict):
    """Test that out-of-range values produce specific error messages"""
    options = SharedOptions()

    with pytest.raises(ValueError) as exc_info:
        options.validate_logit_bias(logit_bias_dict)

    error_msg = str(exc_info.value)

    # The bug: we expect specific error messages but get generic ones
    # This assertion will fail due to the bug
    assert "Value must be between -100 and 100" in error_msg or \
           "Invalid key-value pair" in error_msg

    # Print what we found for demonstration
    print(f"Input: {logit_bias_dict}")
    print(f"Error: {error_msg}")

    # The actual behavior (bug): always get generic message
    assert error_msg == "Invalid key-value pair in logit_bias dictionary"

if __name__ == "__main__":
    # Run the test
    test_validate_logit_bias_error_messages()