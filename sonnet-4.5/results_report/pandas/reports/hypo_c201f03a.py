"""Hypothesis test to detect the ARGSORT_DEFAULTS duplicate key bug"""

import hypothesis
from hypothesis import given, strategies as st, settings
from pandas.compat.numpy.function import ARGSORT_DEFAULTS, ARGSORT_DEFAULTS_KIND

def test_argsort_defaults_duplicate_key():
    """
    Test that verifies ARGSORT_DEFAULTS dictionary has expected keys and values.

    This test detects the duplicate key assignment bug where 'kind' is first
    set to 'quicksort' (line 138) and then immediately overwritten with None (line 140).
    """
    # Verify 'kind' key exists
    assert "kind" in ARGSORT_DEFAULTS, "ARGSORT_DEFAULTS should contain 'kind' key"

    # The current value is None due to line 140 overwriting line 138
    assert ARGSORT_DEFAULTS["kind"] is None, f"Expected None, got {ARGSORT_DEFAULTS['kind']!r}"

    # Verify that ARGSORT_DEFAULTS_KIND intentionally omits 'kind'
    assert "kind" not in ARGSORT_DEFAULTS_KIND, "ARGSORT_DEFAULTS_KIND should not contain 'kind'"

    # Check other expected keys
    assert ARGSORT_DEFAULTS["axis"] == -1
    assert ARGSORT_DEFAULTS["order"] is None
    assert ARGSORT_DEFAULTS["stable"] is None

    print("Test passed: ARGSORT_DEFAULTS['kind'] is None (line 140 overwrites line 138)")
    print("This is a bug: line 138 sets 'kind' to 'quicksort' but is immediately overwritten")

# Run the test
if __name__ == "__main__":
    test_argsort_defaults_duplicate_key()
    print("\nThe test passes, confirming the bug exists:")
    print("- Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'")
    print("- Line 140: ARGSORT_DEFAULTS['kind'] = None  (overwrites line 138)")
    print("\nThis duplicate assignment is clearly a programming error.")