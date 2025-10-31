import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from starlette.middleware.cors import CORSMiddleware

def dummy_app(scope, receive, send):
    pass

@given(st.lists(st.sampled_from(["Accept", "accept", "ACCEPT", "content-type", "Content-Type", "X-Custom-Header", "x-custom-header"]), max_size=10))
@settings(max_examples=500)
def test_cors_no_duplicate_headers(allow_headers):
    middleware = CORSMiddleware(dummy_app, allow_headers=allow_headers)

    unique_headers = set(middleware.allow_headers)
    assert len(middleware.allow_headers) == len(unique_headers), \
        f"Duplicate headers found! Input: {allow_headers}, Output: {middleware.allow_headers}"

if __name__ == "__main__":
    test_cors_no_duplicate_headers()
    print("Test completed")