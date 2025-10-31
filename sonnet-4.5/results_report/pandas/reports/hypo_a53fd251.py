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
    """Test that deprecate rejects docstrings without a blank line after the summary."""
    def bad_alternative():
        pass

    # Create a malformed docstring with no blank line after the summary
    bad_alternative.__doc__ = f"\n{summary}\n{non_empty_after}\n{rest}"

    # This should raise an AssertionError because there's no blank line after the summary
    with pytest.raises(AssertionError):
        deprecate("old", bad_alternative, "1.0")


if __name__ == "__main__":
    print("Running Hypothesis test for deprecate function...")
    print("Testing that deprecate rejects docstrings without a blank line after the summary")
    print("-" * 70)

    try:
        test_deprecate_rejects_malformed_docstrings()
        print("\nAll tests passed! The deprecate function correctly rejects malformed docstrings.")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        print("\nThe deprecate function is NOT properly rejecting malformed docstrings.")
        print("This confirms the bug: docstrings without a blank line after the summary")
        print("are being incorrectly accepted.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()