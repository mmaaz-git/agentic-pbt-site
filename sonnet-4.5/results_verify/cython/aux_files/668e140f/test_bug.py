#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from distutils.dist import Distribution
from Cython.Distutils import build_ext

@settings(max_examples=100)
@given(st.text(min_size=1, max_size=100))
@example("boundscheck=True")
def test_directives_type_validation(directive_value):
    """
    Property: finalize_options should ensure cython_directives is a dict
    """
    dist = Distribution()
    cmd = build_ext(dist)
    cmd.initialize_options()

    cmd.cython_directives = directive_value
    cmd.finalize_options()

    assert isinstance(cmd.cython_directives, dict), \
        f"Expected dict, got {type(cmd.cython_directives)}"

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_directives_type_validation()
        print("Test completed!")
    except Exception as e:
        print(f"Test failed with error: {e}")