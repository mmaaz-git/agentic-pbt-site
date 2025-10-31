#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, Verbosity
import Cython.Compiler.PyrexTypes as PT

@given(st.text())
@settings(max_examples=1000, verbosity=Verbosity.verbose)
def test_escape_type_length_invariant(s):
    result = PT._escape_special_type_characters(s)
    assert len(result) <= 64, f"Length {len(result)} exceeds 64 for input: {repr(s)}"

if __name__ == "__main__":
    # Run the test
    test_escape_type_length_invariant()