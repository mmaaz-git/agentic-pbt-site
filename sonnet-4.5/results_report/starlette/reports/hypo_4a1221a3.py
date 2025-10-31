from hypothesis import given, strategies as st
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware


@given(
    origin=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz:/.-"),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
def test_cors_allow_origins_whitespace(origin, spaces_before, spaces_after):
    origin_with_spaces = " " * spaces_before + origin + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=[origin_with_spaces]
    )

    request_headers = Headers({
        "origin": origin,
        "access-control-request-method": "GET"
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Origin '{origin_with_spaces}' (with spaces) was allowed in config, " \
        f"but request origin '{origin}' (without spaces) was rejected."

if __name__ == "__main__":
    test_cors_allow_origins_whitespace()