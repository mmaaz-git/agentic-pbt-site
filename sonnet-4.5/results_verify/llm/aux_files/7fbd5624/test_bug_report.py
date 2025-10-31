#!/usr/bin/env python3
"""Test script to reproduce the validate_logit_bias exception handling bug"""

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import pytest
from llm.default_plugins.openai_models import SharedOptions
import json

# First let's test the hypothesis test from the bug report
@given(
    st.dictionaries(
        st.integers(min_value=0, max_value=100000).map(str),
        st.integers().filter(lambda x: x < -100 or x > 100),
        min_size=1
    )
)
def test_validate_logit_bias_error_messages(logit_bias_dict):
    options = SharedOptions()

    with pytest.raises(ValueError) as exc_info:
        options.validate_logit_bias(logit_bias_dict)

    error_msg = str(exc_info.value)
    print(f"Testing dict: {logit_bias_dict}, Error: {error_msg}")
    assert "Value must be between -100 and 100" in error_msg or \
           "Invalid key-value pair" in error_msg

# Manual reproduction test
def test_manual_reproduction():
    print("\n=== Manual Reproduction Test ===")
    options = SharedOptions()

    # Test with value out of range (150 > 100)
    print("\nTest 1: Value out of range (150)")
    try:
        result = options.validate_logit_bias({"100": 150})
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Error message contains 'Value must be between -100 and 100': {'Value must be between -100 and 100' in str(e)}")
        print(f"Error message contains 'Invalid key-value pair': {'Invalid key-value pair' in str(e)}")

    # Test with value out of range (negative)
    print("\nTest 2: Value out of range (-150)")
    try:
        result = options.validate_logit_bias({"100": -150})
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Error: {e}")

    # Test with valid value
    print("\nTest 3: Valid value (50)")
    try:
        result = options.validate_logit_bias({"100": 50})
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Error: {e}")

    # Test with invalid key
    print("\nTest 4: Invalid key ('abc')")
    try:
        result = options.validate_logit_bias({"abc": 50})
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Error: {e}")

    # Test with invalid value type
    print("\nTest 5: Invalid value type ('fifty')")
    try:
        result = options.validate_logit_bias({"100": "fifty"})
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run manual tests
    test_manual_reproduction()

    # Run hypothesis tests
    print("\n=== Running Hypothesis Tests ===")
    test_validate_logit_bias_error_messages()