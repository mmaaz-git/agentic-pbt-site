#!/usr/bin/env python3
"""Test script to reproduce the not_nulls bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

# First test: Direct call to not_nulls with a dict
print("Test 1: Direct call to not_nulls with a dict")
print("=" * 50)

from llm.default_plugins.openai_models import not_nulls

options_dict = {"temperature": 0.7, "max_tokens": None, "top_p": 0.9}
print(f"Input dict: {options_dict}")

try:
    result = not_nulls(options_dict)
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print()

# Test 2: Direct call with items()
print("Test 2: Direct call to not_nulls with dict.items()")
print("=" * 50)
print(f"Input dict.items(): {list(options_dict.items())}")

try:
    result = not_nulls(options_dict.items())
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print()

# Test 3: Hypothesis test
print("Test 3: Hypothesis property-based test")
print("=" * 50)

from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(st.dictionaries(
    st.text(min_size=1, max_size=20),
    st.one_of(st.none(), st.integers(), st.floats(allow_nan=False), st.text()),
    min_size=1
))
def test_not_nulls_fails_with_dict_argument(options_dict):
    try:
        result = not_nulls(options_dict)
        assert False, f"Expected ValueError but got result: {result}"
    except (ValueError, TypeError) as e:
        assert "unpack" in str(e) or "iterable" in str(e).lower()
        return True
    except Exception as e:
        assert False, f"Got unexpected error: {type(e).__name__}: {e}"

# Run a few examples
print("Running property test with a few examples...")
test_examples = [
    {"a": 1, "b": None},
    {"x": "test", "y": 42, "z": None},
    {"temp": 0.5},
]

for example in test_examples:
    print(f"Testing with: {example}")
    try:
        test_not_nulls_fails_with_dict_argument(example)
        print("  -> Test passed (ValueError raised as expected)")
    except AssertionError as e:
        print(f"  -> Test failed: {e}")
    except Exception as e:
        print(f"  -> Unexpected error: {type(e).__name__}: {e}")