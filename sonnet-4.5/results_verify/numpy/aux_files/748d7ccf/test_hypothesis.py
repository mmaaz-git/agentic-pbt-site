#!/usr/bin/env python3
"""Hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
import numpy.ctypeslib

@settings(max_examples=200)
@given(st.lists(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ_', min_size=0, max_size=20), min_size=1, max_size=5))
def test_comma_separated_invalid_flags_error_type(flags_list):
    flag_str = ','.join(flags_list)

    try:
        ptr = numpy.ctypeslib.ndpointer(flags=flag_str)
    except TypeError:
        pass
    except KeyError as e:
        if str(e) == "''":
            assert False, f"BUG: Got unhelpful KeyError('') instead of TypeError for flags '{flag_str}'"

# Run the test
print("Running hypothesis test...")
test_comma_separated_invalid_flags_error_type()
print("Test completed without finding the bug!")