import warnings
from hypothesis import given, strategies as st, settings
import pytest
import numpy.typing as npt


def test_nbitbase_access_emits_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "NBitBase" in str(w[0].message)
        assert "deprecated" in str(w[0].message).lower()

if __name__ == "__main__":
    try:
        test_nbitbase_access_emits_deprecation()
        print("Test passed: Deprecation warning was emitted correctly")
    except AssertionError as e:
        print(f"Test failed: {e}")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = npt.NBitBase
            print(f"Number of warnings: {len(w)}")
            if len(w) > 0:
                for warning in w:
                    print(f"  Warning: {warning.category.__name__}: {warning.message}")