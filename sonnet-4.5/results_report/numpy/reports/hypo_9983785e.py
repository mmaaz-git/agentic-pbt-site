import warnings
import sys
from hypothesis import given, strategies as st


def fresh_import_numpy_typing():
    """Ensure a fresh import of numpy.typing module"""
    if 'numpy.typing' in sys.modules:
        del sys.modules['numpy.typing']
    import numpy.typing
    return numpy.typing


@given(st.just(None))  # Using a dummy strategy since we're not testing with varying inputs
def test_nbits_deprecation_warning(dummy):
    """Test that accessing NBitBase triggers a DeprecationWarning"""
    npt = fresh_import_numpy_typing()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Access the supposedly deprecated NBitBase
        result = npt.NBitBase

        # Verify a deprecation warning was triggered
        assert len(w) > 0, "Expected DeprecationWarning when accessing NBitBase"
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w), \
            f"Expected DeprecationWarning but got: {[warning.category for warning in w]}"

        # Check the warning message content
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0, "No DeprecationWarning found"

        warning_msg = str(deprecation_warnings[0].message)
        assert "NBitBase" in warning_msg, f"Warning message should mention NBitBase: {warning_msg}"
        assert "deprecated" in warning_msg.lower(), f"Warning should mention deprecation: {warning_msg}"


if __name__ == "__main__":
    # Run the test
    test_nbits_deprecation_warning()