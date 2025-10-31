#!/usr/bin/env python3
"""Property-based test from the bug report."""

import os
import sys

# Add Flask environment to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/flask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from flask.helpers import get_debug_flag, get_load_dotenv


@given(
    st.sampled_from(["false", "False", "no", "No", "0"]),
    st.sampled_from([" ", "\t", "  "])
)
def test_debug_flag_whitespace_stripping(value, whitespace):
    os.environ['FLASK_DEBUG'] = whitespace + value
    result_leading = get_debug_flag()

    os.environ['FLASK_DEBUG'] = value
    expected = get_debug_flag()

    assert result_leading == expected, (
        f"Leading whitespace changed result: {whitespace + value!r} → {result_leading}, "
        f"expected {expected}"
    )


@given(
    st.sampled_from(["false", "False", "no", "No", "0"]),
    st.sampled_from([" ", "\t", "  "])
)
def test_load_dotenv_whitespace_stripping(value, whitespace):
    os.environ['FLASK_SKIP_DOTENV'] = whitespace + value
    result_leading = get_load_dotenv(True)

    os.environ['FLASK_SKIP_DOTENV'] = value
    expected = get_load_dotenv(True)

    assert result_leading == expected, (
        f"Leading whitespace changed result: {whitespace + value!r} → {result_leading}, "
        f"expected {expected}"
    )

# Run the tests
if __name__ == "__main__":
    print("Testing get_debug_flag with Hypothesis...")
    try:
        test_debug_flag_whitespace_stripping()
        print("✓ get_debug_flag test passed (unexpected - bug should be present)")
    except AssertionError as e:
        print(f"✗ get_debug_flag test failed as expected: {e}")

    print("\nTesting get_load_dotenv with Hypothesis...")
    try:
        test_load_dotenv_whitespace_stripping()
        print("✓ get_load_dotenv test passed (unexpected - bug should be present)")
    except AssertionError as e:
        print(f"✗ get_load_dotenv test failed as expected: {e}")

    # Cleanup
    if 'FLASK_DEBUG' in os.environ:
        del os.environ['FLASK_DEBUG']
    if 'FLASK_SKIP_DOTENV' in os.environ:
        del os.environ['FLASK_SKIP_DOTENV']