"""Proper hypothesis test that will fail on the bug"""

from hypothesis import given, settings, strategies as st, assume
from starlette.middleware.cors import CORSMiddleware


def dummy_app(scope, receive, send):
    pass


@given(st.lists(st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1)))
@settings(max_examples=1000)
def test_cors_allow_headers_no_duplicates(headers):
    """Test that CORSMiddleware doesn't create duplicate headers"""
    middleware = CORSMiddleware(dummy_app, allow_headers=headers)

    # This assertion should always be true - no duplicates should exist
    assert len(middleware.allow_headers) == len(set(middleware.allow_headers)), \
        f"Duplicates found! Input: {headers}, Output: {middleware.allow_headers}"


# Run the test
if __name__ == "__main__":
    try:
        test_cors_allow_headers_no_duplicates()
        print("Test passed - no duplicates found")
    except AssertionError as e:
        print(f"Test FAILED: {e}")

    # Also test the minimal failing case
    print("\nMinimal failing case:")
    try:
        test_cors_allow_headers_no_duplicates.function(['a', 'A'])
    except AssertionError as e:
        print(f"As expected, failed with: {e}")