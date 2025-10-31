from hypothesis import given, strategies as st, settings
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware


@given(
    header_name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz-"),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
@settings(max_examples=100)
def test_cors_allow_headers_whitespace(header_name, spaces_before, spaces_after):
    header_with_spaces = " " * spaces_before + header_name + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=["*"],
        allow_headers=[header_with_spaces]
    )

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": "GET",
        "access-control-request-headers": header_name
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Header '{header_with_spaces}' (with spaces) was allowed in config, " \
        f"but request header '{header_name}' (without spaces) was rejected."


# Run the test
if __name__ == "__main__":
    print("Running property-based test for CORS header whitespace handling...")
    try:
        test_cors_allow_headers_whitespace()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()