import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st
import pytest


def test_nbitbase_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "NBitBase" in str(w[0].message)

if __name__ == "__main__":
    test_nbitbase_deprecation_warning()
    print("Test passed!")