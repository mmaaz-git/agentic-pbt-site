import warnings
from hypothesis import given, strategies as st, settings
import numpy.typing as npt


@given(st.none())
@settings(max_examples=1)
def test_nbitbase_access_emits_deprecation(_):
    """Test that accessing npt.NBitBase emits a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase

        assert len(w) == 1, f"Expected 1 deprecation warning, got {len(w)}"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category}"
        assert "NBitBase" in str(w[0].message), f"Warning message doesn't mention NBitBase: {w[0].message}"
        assert "deprecated" in str(w[0].message).lower(), f"Warning message doesn't mention deprecation: {w[0].message}"


if __name__ == "__main__":
    # Run the test
    try:
        test_nbitbase_access_emits_deprecation()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")