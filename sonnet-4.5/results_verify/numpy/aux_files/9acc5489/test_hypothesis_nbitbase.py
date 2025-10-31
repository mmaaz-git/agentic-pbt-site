import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_nbitbase_emits_deprecation_warning(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)

        assert len(w) >= 1, "Expected deprecation warning to be emitted"
        assert issubclass(w[0].category, DeprecationWarning)
        assert "NBitBase" in str(w[0].message)
        assert "deprecated" in str(w[0].message).lower()

# Run the test
try:
    test_nbitbase_emits_deprecation_warning()
    print("Test passed unexpectedly!")
except AssertionError as e:
    print(f"Test failed as expected: {e}")