import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_deprecated_attribute_warns(attr_name):
    """Test that accessing NBitBase triggers a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        getattr(npt, attr_name)
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category}"

# Run the test
if __name__ == "__main__":
    test_deprecated_attribute_warns()