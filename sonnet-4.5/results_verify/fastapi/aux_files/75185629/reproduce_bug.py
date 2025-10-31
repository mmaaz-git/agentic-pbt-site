import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from starlette.middleware.cors import CORSMiddleware

def dummy_app(scope, receive, send):
    pass

middleware = CORSMiddleware(dummy_app, allow_headers=["accept"])

print("middleware.allow_headers =", middleware.allow_headers)
print("Length of allow_headers:", len(middleware.allow_headers))
print("Length of unique headers:", len(set(middleware.allow_headers)))

try:
    assert len(middleware.allow_headers) == len(set(middleware.allow_headers))
    print("No duplicates found")
except AssertionError:
    print("AssertionError: Duplicates found!")