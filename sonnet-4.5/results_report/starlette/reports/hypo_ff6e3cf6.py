from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def dummy_app(scope, receive, send):
    pass


@given(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
def test_cors_header_whitespace_consistency(header_name):
    cors_with_spaces = CORSMiddleware(
        app=dummy_app,
        allow_origins=["https://example.com"],
        allow_headers=[f" {header_name} "],
    )

    cors_without_spaces = CORSMiddleware(
        app=dummy_app,
        allow_origins=["https://example.com"],
        allow_headers=[header_name],
    )

    headers_requesting = Headers(raw=[
        (b"origin", b"https://example.com"),
        (b"access-control-request-method", b"GET"),
        (b"access-control-request-headers", header_name.encode()),
    ])

    response_without_spaces = cors_without_spaces.preflight_response(request_headers=headers_requesting)
    assert response_without_spaces.status_code == 200

    response_with_spaces = cors_with_spaces.preflight_response(request_headers=headers_requesting)
    assert response_with_spaces.status_code == 200, "Headers with spaces in config should still match"


if __name__ == "__main__":
    test_cors_header_whitespace_consistency()