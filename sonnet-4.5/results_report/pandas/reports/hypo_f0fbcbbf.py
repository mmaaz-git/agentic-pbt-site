from hypothesis import given, strategies as st
from pandas.core.computation.common import ensure_decoded

@given(st.binary())
def test_ensure_decoded_returns_str(data):
    result = ensure_decoded(data)
    assert isinstance(result, str)

if __name__ == "__main__":
    test_ensure_decoded_returns_str()