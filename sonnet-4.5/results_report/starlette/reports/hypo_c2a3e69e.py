from hypothesis import given, strategies as st
from starlette.datastructures import URL

@given(
    st.sampled_from(["http", "https", "ws", "wss"]),
    st.text(min_size=1),
    st.integers(min_value=1, max_value=65535) | st.none()
)
def test_url_replace_with_empty_netloc(scheme, path, port):
    url_str = f"{scheme}:///{path}"
    url = URL(url_str)

    if port is not None:
        new_url = url.replace(port=port)
        assert new_url.port == port

if __name__ == "__main__":
    test_url_replace_with_empty_netloc()