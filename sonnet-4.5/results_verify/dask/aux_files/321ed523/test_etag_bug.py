from hypothesis import given, strategies as st
from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles

@given(st.text(min_size=1, max_size=20))
def test_etag_weak_and_strong_match(etag_value):
    response_headers = Headers({"etag": f'W/"{etag_value}"'})
    request_headers = Headers({"if-none-match": f'W/"{etag_value}"'})

    static_files = StaticFiles(directory="/tmp")

    result = static_files.is_not_modified(response_headers, request_headers)
    assert result is True, f"Weak ETag W/\"{etag_value}\" should match itself"

# Run the test
if __name__ == "__main__":
    test_etag_weak_and_strong_match()
    print("Test completed - if no assertion error, test passed")