import io
import warnings
from hypothesis import given, strategies as st, settings
from scipy.io import savemat
from scipy.io.matlab import MatWriteWarning


@settings(max_examples=100)
@given(
    invalid_key=st.one_of(
        st.text(min_size=1, max_size=10).filter(lambda s: s[0] == '_'),
        st.text(min_size=1, max_size=10).filter(lambda s: s[0].isdigit()),
    ),
    value=st.integers(min_value=0, max_value=100)
)
def test_savemat_invalid_key_warning(invalid_key, value):
    file_obj = io.BytesIO()
    data = {invalid_key: value}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(file_obj, data)

        assert len(w) > 0, f"No warning issued for invalid key {invalid_key}"
        assert any(issubclass(warning.category, MatWriteWarning) for warning in w), \
            f"Expected MatWriteWarning for invalid key {invalid_key}"

if __name__ == "__main__":
    # Run the test
    test_savemat_invalid_key_warning()