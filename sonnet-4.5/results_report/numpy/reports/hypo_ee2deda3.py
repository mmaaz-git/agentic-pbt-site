import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st


def test_nbitbase_deprecation():
    """Test that accessing NBitBase emits a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category}"
        assert "NBitBase" in str(w[0].message), f"'NBitBase' not found in warning message: {w[0].message}"


if __name__ == "__main__":
    # Run the test
    test_nbitbase_deprecation()
    print("Test passed: NBitBase deprecation warning was emitted successfully")