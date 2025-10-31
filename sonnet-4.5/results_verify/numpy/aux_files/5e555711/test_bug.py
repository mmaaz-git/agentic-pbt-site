import warnings
import sys
from hypothesis import given, strategies as st


def fresh_import_numpy_typing():
    if 'numpy.typing' in sys.modules:
        del sys.modules['numpy.typing']
    import numpy.typing
    return numpy.typing


def test_nbits_deprecation_warning():
    npt = fresh_import_numpy_typing()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = npt.NBitBase

        assert len(w) > 0, "Expected DeprecationWarning when accessing NBitBase"
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w), \
            f"Expected DeprecationWarning but got: {[warning.category for warning in w]}"

# Run the test
print("Running Hypothesis test:")
try:
    test_nbits_deprecation_warning()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")