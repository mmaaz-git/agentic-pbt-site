from starlette.middleware.cors import CORSMiddleware


def dummy_app(scope, receive, send):
    pass


cors = CORSMiddleware(
    dummy_app,
    allow_origins=["*"],
    allow_headers=["accept"],
    allow_methods=["GET"]
)

print("allow_headers:", cors.allow_headers)
print("Expected: ['accept', 'accept-language', 'content-language', 'content-type']")
print("Actual has duplicate 'accept':", cors.allow_headers)
print("Length of allow_headers:", len(cors.allow_headers))
print("Length of set(allow_headers):", len(set(cors.allow_headers)))
assert len(cors.allow_headers) == len(set(cors.allow_headers)), f"Duplicate headers found: {cors.allow_headers}"