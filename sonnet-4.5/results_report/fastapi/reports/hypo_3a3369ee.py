from hypothesis import given, strategies as st, settings, example
from starlette.middleware.cors import CORSMiddleware

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=20)))
@settings(max_examples=1000)
@example(['Q', 'q'])  # Known failing example
def test_cors_allow_headers_no_duplicates(allow_headers):
    async def dummy_app(scope, receive, send):
        pass

    middleware = CORSMiddleware(app=dummy_app, allow_headers=allow_headers)
    assert len(middleware.allow_headers) == len(set(middleware.allow_headers))

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=20)))
@settings(max_examples=1000)
@example(['['])  # Known failing example
def test_cors_allow_headers_sorted(allow_headers):
    async def dummy_app(scope, receive, send):
        pass

    middleware = CORSMiddleware(app=dummy_app, allow_headers=allow_headers)
    assert middleware.allow_headers == sorted(middleware.allow_headers)

# Run the tests
if __name__ == "__main__":
    import traceback

    print("Running property-based tests for CORSMiddleware...")
    print("\nTest 1: Check for no duplicates")
    try:
        test_cors_allow_headers_no_duplicates()
        print("✓ No duplicates test passed")
    except Exception as e:
        print("✗ Duplicates test failed")
        traceback.print_exc()

    print("\nTest 2: Check for proper sorting")
    try:
        test_cors_allow_headers_sorted()
        print("✓ Sorting test passed")
    except Exception as e:
        print("✗ Sorting test failed")
        traceback.print_exc()