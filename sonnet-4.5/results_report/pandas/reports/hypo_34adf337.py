from hypothesis import given, strategies as st
from pandas.io.common import is_url


@given(st.text())
def test_is_url_handles_arbitrary_input(url_path):
    full_url = f"http://{url_path}"
    result = is_url(full_url)
    assert isinstance(result, bool)

if __name__ == "__main__":
    test_is_url_handles_arbitrary_input()