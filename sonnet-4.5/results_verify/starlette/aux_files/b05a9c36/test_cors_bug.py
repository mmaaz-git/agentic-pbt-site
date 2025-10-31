"""Test to reproduce the reported CORS duplicate headers bug"""

from hypothesis import given, settings, strategies as st
from starlette.middleware.cors import CORSMiddleware, SAFELISTED_HEADERS


def dummy_app(scope, receive, send):
    pass


# First, let's run the specific failing examples mentioned
print("=== Testing specific failing examples ===")

# Test case 1: ['F', 'f']
middleware1 = CORSMiddleware(dummy_app, allow_headers=['F', 'f'])
print(f"Input: ['F', 'f']")
print(f"Result allow_headers: {middleware1.allow_headers}")
print(f"Length: {len(middleware1.allow_headers)}, Unique: {len(set(middleware1.allow_headers))}")
print(f"Has duplicates: {len(middleware1.allow_headers) != len(set(middleware1.allow_headers))}")
print()

# Test case 2: ['X-Custom-Header', 'x-custom-header']
middleware2 = CORSMiddleware(dummy_app, allow_headers=['X-Custom-Header', 'x-custom-header'])
print(f"Input: ['X-Custom-Header', 'x-custom-header']")
print(f"Result allow_headers: {middleware2.allow_headers}")
print(f"Length: {len(middleware2.allow_headers)}, Unique: {len(set(middleware2.allow_headers))}")
print(f"Has duplicates: {len(middleware2.allow_headers) != len(set(middleware2.allow_headers))}")
print()

# Now run the property-based test
print("=== Running property-based test ===")

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=1, blacklist_categories=('Cc', 'Cs')))))
@settings(max_examples=100)  # Reduced for faster testing
def test_cors_allow_headers_no_duplicates(headers):
    middleware = CORSMiddleware(dummy_app, allow_headers=headers)
    if len(middleware.allow_headers) != len(set(middleware.allow_headers)):
        print(f"FAILURE: Input headers: {headers}")
        print(f"Result allow_headers: {middleware.allow_headers}")
        print(f"Duplicates found!")
        return False
    return True

# Run the property test
try:
    test_cors_allow_headers_no_duplicates()
    print("Property test passed (no failures found in 100 examples)")
except Exception as e:
    print(f"Property test failed with error: {e}")

# Let's also manually check what happens internally
print("\n=== Detailed analysis of the bug ===")
test_headers = ['X-Custom', 'x-custom']
print(f"Input headers: {test_headers}")
print(f"SAFELISTED_HEADERS: {SAFELISTED_HEADERS}")
print(f"set(test_headers): {set(test_headers)}")
print(f"SAFELISTED_HEADERS | set(test_headers): {SAFELISTED_HEADERS | set(test_headers)}")
sorted_headers = sorted(SAFELISTED_HEADERS | set(test_headers))
print(f"After sorted(): {sorted_headers}")
lowercased = [h.lower() for h in sorted_headers]
print(f"After lowercasing: {lowercased}")
print(f"Duplicates in final list: {[h for h in lowercased if lowercased.count(h) > 1]}")