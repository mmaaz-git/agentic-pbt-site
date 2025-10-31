from hypothesis import given, strategies as st
from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles

@given(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126)).filter(lambda x: '"' not in x and ',' not in x))
def test_etag_weak_and_strong_match(etag_value):
    # Test that a weak ETag matches itself
    response_headers = Headers({"etag": f'W/"{etag_value}"'})
    request_headers = Headers({"if-none-match": f'W/"{etag_value}"'})

    static_files = StaticFiles(directory="/tmp", check_dir=False)
    result = static_files.is_not_modified(response_headers, request_headers)

    assert result is True, f"Weak ETag W/\"{etag_value}\" should match itself"

if __name__ == "__main__":
    # Run the test
    test_etag_weak_and_strong_match()