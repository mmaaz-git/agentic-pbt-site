from hypothesis import given, settings, strategies as st
from pandas.core.dtypes.common import ensure_str

@settings(max_examples=1000)
@given(
    st.one_of(
        st.binary(),
        st.text(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
    )
)
def test_ensure_str_returns_str(value):
    result = ensure_str(value)
    assert isinstance(result, str), f"Expected str, got {type(result)}"

if __name__ == "__main__":
    test_ensure_str_returns_str()