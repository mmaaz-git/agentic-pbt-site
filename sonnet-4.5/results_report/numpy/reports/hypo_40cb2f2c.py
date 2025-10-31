import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st


@given(st.just("NBitBase"))
def test_nbitbase_emits_deprecation_warning(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)

        assert len(w) >= 1, f"Expected deprecation warning for {attr_name}"
        assert any("deprecated" in str(w_item.message).lower() for w_item in w)


# Run the test
test_nbitbase_emits_deprecation_warning()