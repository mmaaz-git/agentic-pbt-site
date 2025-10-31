import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_deprecated_attributes_trigger_warnings(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)
        assert len(w) >= 1, f"Expected deprecation warning for {attr_name} but got none"
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

# Run the test
test_deprecated_attributes_trigger_warnings()