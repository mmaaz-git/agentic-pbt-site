#!/usr/bin/env python3
import sys
import string
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Tempita import sub

@given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=20))
@settings(max_examples=100)
def test_sub_name_parameter_isolation(template_name):
    content = "{{__name}}"

    result = sub(content, __name=template_name)

    assert result == '', f"__name should not be accessible in template namespace, but got: {result}"

# Run the test
try:
    test_sub_name_parameter_isolation()
    print("Hypothesis test passed - no failures found")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")