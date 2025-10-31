#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

print("Testing the cmp_using error message bug...")
print("-" * 50)

# Test 1: Direct reproduction as shown in bug report
from attr._cmp import cmp_using

try:
    cmp_using(lt=lambda a, b: a < b)
except ValueError as e:
    print(f"Error message: {e}")
    print()

# Test 2: Try with hypothesis test
from hypothesis import given, strategies as st

@given(st.sampled_from([lambda a, b: a < b, lambda a, b: a > b]))
def test_cmp_using_error_message_grammar(lt_func):
    try:
        cmp_using(lt=lt_func)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        print(f"Hypothesis test error message: {error_msg}")
        # Check if the message has the typos
        if "define is order" in error_msg:
            print("BUG CONFIRMED: Error message contains typos")
            print("  - 'define' should be 'defined'")
            print("  - 'is order' should be 'in order'")
        else:
            print("Unexpected: Error message doesn't match expected typo pattern")

# Run the hypothesis test
print("Running hypothesis test...")
test_cmp_using_error_message_grammar()