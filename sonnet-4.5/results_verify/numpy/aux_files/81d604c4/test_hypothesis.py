import warnings
from hypothesis import given, strategies as st


@given(st.just("NBitBase"))
def test_nbitbase_deprecation_warning(attr_name):
    import numpy.typing as npt

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)

        assert len(w) >= 1, f"Expected DeprecationWarning when accessing {attr_name}, but got no warnings"
        assert any(
            issubclass(warning.category, DeprecationWarning) and "NBitBase" in str(warning.message)
            for warning in w
        ), f"Expected DeprecationWarning about NBitBase, but got: {[str(w_.message) for w_ in w]}"


# Run the test
if __name__ == "__main__":
    test_nbitbase_deprecation_warning()
    print("Hypothesis test passed!")