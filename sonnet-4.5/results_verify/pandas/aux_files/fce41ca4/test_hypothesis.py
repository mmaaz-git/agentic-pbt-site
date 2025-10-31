from hypothesis import given, strategies as st, settings
from pandas.util._decorators import deprecate
import pytest


@given(
    st.text(min_size=1),
    st.text(min_size=1),
    st.text(min_size=1),
)
@settings(max_examples=10)
def test_deprecate_rejects_malformed_docstrings(summary, non_empty_after, rest):
    def bad_alternative():
        pass

    bad_alternative.__doc__ = f"\n{summary}\n{non_empty_after}\n{rest}"

    with pytest.raises(AssertionError):
        deprecate("old", bad_alternative, "1.0")
    print(f"  Tested: summary='{summary[:10]}...', non_empty='{non_empty_after[:10]}...'")


print("Running hypothesis test...")
try:
    test_deprecate_rejects_malformed_docstrings()
    print("Test completed without failures")
except Exception as e:
    print(f"Test failed: {e}")

# Run a specific failing example
print("\nSpecific failing example:")
def bad():
    pass
bad.__doc__ = "\nSummary\nNext line\nRest"

try:
    deprecate("old", bad, "1.0")
    print("BUG CONFIRMED: Malformed docstring accepted when it should be rejected")
except AssertionError:
    print("Docstring correctly rejected")