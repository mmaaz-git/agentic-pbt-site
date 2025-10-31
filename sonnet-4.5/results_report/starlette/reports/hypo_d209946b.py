import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def dummy_app(scope, receive, send):
    pass


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


if __name__ == "__main__":
    test_cors_whitespace_only_header()