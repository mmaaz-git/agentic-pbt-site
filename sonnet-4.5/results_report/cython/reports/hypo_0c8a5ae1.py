#!/usr/bin/env python3
"""
Hypothesis test that discovered the Cython.Plex.Regexps.RE.wrong_type bug
"""

import pytest
from hypothesis import given, strategies as st, settings
from Cython.Plex import Seq, Str
from Cython.Plex.Errors import PlexTypeError

@given(st.text(alphabet='abc', min_size=1, max_size=5))
@settings(max_examples=200)
def test_seq_rejects_non_re_args(s):
    """Test that Seq properly rejects non-RE arguments with PlexTypeError"""
    with pytest.raises(PlexTypeError):
        # The second argument to Seq should be an RE, not a string
        # This should raise PlexTypeError with a helpful message
        Seq(Str(s), "not an RE")

if __name__ == "__main__":
    # Run the test manually without Hypothesis decoration
    try:
        with pytest.raises(PlexTypeError):
            Seq(Str('a'), "not an RE")
        print("Test passed (PlexTypeError was raised as expected)")
    except AttributeError as e:
        print(f"Test failed with AttributeError: {e}")
        print("\nExpected: PlexTypeError with message about invalid type")
        print("Actual: AttributeError because types.InstanceType doesn't exist in Python 3")
        print("\nFailing input: s='a' (or any string value)")