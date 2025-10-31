from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware


@given(
    header=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"), whitelist_characters="-"))
)
def test_cors_headers_case_insensitive_property(header):
    middleware_upper = CORSMiddleware(None, allow_headers=[header.upper()])
    middleware_lower = CORSMiddleware(None, allow_headers=[header.lower()])

    assert middleware_upper.allow_headers == middleware_lower.allow_headers


# Run the test
if __name__ == "__main__":
    test_cors_headers_case_insensitive_property()