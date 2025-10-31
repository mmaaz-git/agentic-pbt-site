from hypothesis import given, strategies as st
import pandas.io.common as common


@given(scheme=st.sampled_from(['http', 'https', 'ftp', 'file']),
       domain=st.text(min_size=1))
def test_is_url_returns_bool(scheme, domain):
    url = f"{scheme}://{domain}"
    result = common.is_url(url)
    assert isinstance(result, bool)

if __name__ == "__main__":
    test_is_url_returns_bool()