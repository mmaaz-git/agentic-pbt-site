import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st
import pytest


def test_nbitbase_deprecation_warning():
    """Test that accessing NBitBase emits a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase
        assert len(w) == 1, f"Expected 1 DeprecationWarning, got {len(w)} warnings"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category}"
        assert "NBitBase" in str(w[0].message), f"Warning message doesn't mention NBitBase: {w[0].message}"


if __name__ == "__main__":
    # Run the test
    try:
        test_nbitbase_deprecation_warning()
        print("Test PASSED: NBitBase deprecation warning was emitted correctly")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"Test ERROR: {e}")