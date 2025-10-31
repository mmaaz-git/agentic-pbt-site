from hypothesis import given, settings, strategies as st, assume
from starlette.middleware.cors import CORSMiddleware


@given(
    header_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-")),
    leading_space=st.integers(min_value=0, max_value=3),
    trailing_space=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=200)
def test_cors_allow_headers_whitespace_inconsistency(header_name, leading_space, trailing_space):
    assume(header_name.strip() == header_name)
    assume(len(header_name.strip()) > 0)

    header_with_whitespace = " " * leading_space + header_name + " " * trailing_space

    async def dummy_app(scope, receive, send):
        pass

    middleware = CORSMiddleware(
        dummy_app,
        allow_origins=["http://example.com"],
        allow_headers=[header_with_whitespace]
    )

    stored_header = middleware.allow_headers[0]

    assert stored_header.strip() == stored_header, \
        f"Bug: allow_headers contains whitespace: '{stored_header}' (should be '{stored_header.strip()}')"


if __name__ == "__main__":
    # Run the test
    test_cors_allow_headers_whitespace_inconsistency()