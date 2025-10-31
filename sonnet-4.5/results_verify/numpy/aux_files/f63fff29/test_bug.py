#!/usr/bin/env python3
"""Test the bug report about removespaces"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

# First test the hypothesis test
from hypothesis import given, settings
from hypothesis import strategies as st
import numpy.f2py.crackfortran as cf

@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_removespaces_preserves_non_space_whitespace(text):
    result = cf.removespaces(text)
    expected = text.replace(' ', '')

    for char in ['\n', '\r', '\t']:
        if char in text and char not in ' ':
            if char in expected and char not in result:
                assert False, f"removespaces removed {repr(char)} which is not a space"

# Run the hypothesis test
print("Running hypothesis test...")
try:
    test_removespaces_preserves_non_space_whitespace()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Now test the specific examples
print("\nTesting specific examples:")
print(f"removespaces('\\n') = {repr(cf.removespaces('\n'))}")
print(f"removespaces('\\ra\\r') = {repr(cf.removespaces('\ra\r'))}")
print(f"removespaces('\\t') = {repr(cf.removespaces('\t'))}")

print("\nBug report expectations:")
print("Expected: '\\n', '\\ra\\r', '\\t'")
print("Actual: '', 'a', ''")

# Additional tests to understand behavior
print("\nAdditional tests:")
print(f"removespaces('  a  ') = {repr(cf.removespaces('  a  '))}")
print(f"removespaces(' ') = {repr(cf.removespaces(' '))}")
print(f"removespaces('a b c') = {repr(cf.removespaces('a b c'))}")
print(f"removespaces('a + b') = {repr(cf.removespaces('a + b'))}")
print(f"removespaces('(a b)') = {repr(cf.removespaces('(a b)'))}")
print(f"removespaces('\\n a \\n') = {repr(cf.removespaces('\n a \n'))}")
print(f"removespaces('\\t\\ta\\t\\t') = {repr(cf.removespaces('\t\ta\t\t'))}")