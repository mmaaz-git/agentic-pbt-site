#!/usr/bin/env python3
"""Test script to reproduce the reported Unicode header bug in Starlette CORS middleware."""

import sys
import traceback

# Add the path as specified in the bug report
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

print("Testing Unicode header bug in Starlette CORS middleware")
print("=" * 60)

# First, let's try the exact reproduction code from the bug report
print("\n1. Running the exact reproduction code from the bug report:")
print("-" * 40)

try:
    from starlette.middleware.cors import CORSMiddleware
    from starlette.datastructures import Headers

    def dummy_app(scope, receive, send):
        pass

    # Create middleware with non-latin-1 character 'Ā' (U+0100)
    print("Creating CORSMiddleware with allow_headers=['Ā']...")
    middleware = CORSMiddleware(dummy_app, allow_headers=['Ā'], allow_origins=["*"])
    print("Middleware created successfully!")

    # Create request headers
    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": "GET"
    })

    print("Calling preflight_response()...")
    response = middleware.preflight_response(request_headers=request_headers)
    print(f"Response received successfully! Status code: {response.status_code}")

except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError caught: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Other exception caught: {type(e).__name__}: {e}")
    traceback.print_exc()

# Now let's test with a latin-1 encodable character for comparison
print("\n2. Testing with latin-1 encodable header for comparison:")
print("-" * 40)

try:
    middleware2 = CORSMiddleware(dummy_app, allow_headers=['X-Custom-Header'], allow_origins=["*"])
    print("Middleware created with latin-1 compatible header")

    request_headers2 = Headers({
        "origin": "http://example.com",
        "access-control-request-method": "GET"
    })

    response2 = middleware2.preflight_response(request_headers=request_headers2)
    print(f"Response received successfully! Status code: {response2.status_code}")

except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
    traceback.print_exc()

# Let's also test the property-based test from the bug report
print("\n3. Running the property-based test with the failing input:")
print("-" * 40)

try:
    from hypothesis import given, strategies as st, assume

    @given(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
    def test_cors_whitespace_only_header(allowed_headers):
        assume(all(h.strip() for h in allowed_headers))

        middleware = CORSMiddleware(dummy_app, allow_headers=allowed_headers, allow_origins=["*"])

        request_headers = Headers({
            "origin": "http://example.com",
            "access-control-request-method": "GET",
            "access-control-request-headers": "   "
        })

        response = middleware.preflight_response(request_headers=request_headers)
        assert response.status_code in [200, 400]

    # Test with the specific failing input
    print("Testing with allowed_headers=['Ā']...")
    test_cors_whitespace_only_header(['Ā'])
    print("Test passed without error!")

except Exception as e:
    print(f"Exception in property test: {type(e).__name__}: {e}")
    traceback.print_exc()

# Let's test various non-latin-1 characters
print("\n4. Testing various non-latin-1 characters:")
print("-" * 40)

non_latin1_chars = [
    ('Ā', 'U+0100', 'Latin Extended-A'),
    ('Ω', 'U+03A9', 'Greek'),
    ('♥', 'U+2665', 'Symbol'),
    ('你', 'U+4F60', 'Chinese'),
    ('😀', 'U+1F600', 'Emoji'),
]

for char, code, desc in non_latin1_chars:
    try:
        print(f"\nTesting with '{char}' ({code} - {desc}):")
        middleware = CORSMiddleware(dummy_app, allow_headers=[char], allow_origins=["*"])
        request_headers = Headers({
            "origin": "http://example.com",
            "access-control-request-method": "GET"
        })
        response = middleware.preflight_response(request_headers=request_headers)
        print(f"  Success! Status: {response.status_code}")
    except UnicodeEncodeError as e:
        print(f"  UnicodeEncodeError: {e}")
    except Exception as e:
        print(f"  Other error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing complete!")