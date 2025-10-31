#!/usr/bin/env python3
import sys
import os

# Activate the virtual environment's Python path
venv_path = "/home/npc/pbt/agentic-pbt/envs/llm_env"
sys.path.insert(0, os.path.join(venv_path, 'lib', 'python3.13', 'site-packages'))

from hypothesis import given, strategies as st
import llm.utils

@given(
    st.text(min_size=1, max_size=1000),
    st.integers(min_value=1, max_value=500)
)
def test_truncate_string_length_property(text, max_length):
    result = llm.utils.truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"len('{result}') = {len(result)} > max_length={max_length}"

# Run the property-based test
print("Running property-based test...")
try:
    test_truncate_string_length_property()
    print("Test passed! (This shouldn't happen if the bug exists)")
except AssertionError as e:
    print(f"Test failed as expected with assertion error: {e}")