from hypothesis import given, strategies as st, settings
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware


@given(
    method=st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
@settings(max_examples=50)
def test_cors_allow_methods_whitespace(method, spaces_before, spaces_after):
    method_with_spaces = " " * spaces_before + method + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=["*"],
        allow_methods=[method_with_spaces]
    )

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": method
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Method '{method_with_spaces}' (with spaces) was allowed in config, " \
        f"but request method '{method}' (without spaces) was rejected."

if __name__ == "__main__":
    test_cors_allow_methods_whitespace()