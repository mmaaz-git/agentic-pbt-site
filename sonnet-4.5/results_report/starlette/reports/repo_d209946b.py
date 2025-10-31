import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def dummy_app(scope, receive, send):
    pass


# Create middleware with non-latin-1 character in allow_headers
middleware = CORSMiddleware(dummy_app, allow_headers=['Ā'], allow_origins=["*"])

# Create request headers for a preflight request
request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET"
})

# This will raise UnicodeEncodeError
response = middleware.preflight_response(request_headers=request_headers)
print(f"Response status: {response.status_code}")