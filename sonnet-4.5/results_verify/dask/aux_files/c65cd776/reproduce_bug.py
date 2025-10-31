from starlette.middleware.cors import CORSMiddleware, SAFELISTED_HEADERS


def dummy_app(scope, receive, send):
    pass


# First, let's check what SAFELISTED_HEADERS contains
print("SAFELISTED_HEADERS:", SAFELISTED_HEADERS)
print()

# Reproduce the bug as described
cors = CORSMiddleware(
    dummy_app,
    allow_origins=["*"],
    allow_headers=["accept"],  # lowercase version of a safelisted header
    allow_methods=["GET"]
)

print("cors.allow_headers:", cors.allow_headers)
print("Number of headers:", len(cors.allow_headers))
print("Unique headers:", len(set(cors.allow_headers)))
print()

# Check for duplicates
duplicates = []
seen = set()
for header in cors.allow_headers:
    if header in seen:
        duplicates.append(header)
    seen.add(header)

if duplicates:
    print("DUPLICATES FOUND:", duplicates)
else:
    print("No duplicates found")

# Assertion as in the bug report
try:
    assert len(cors.allow_headers) == len(set(cors.allow_headers)), \
        f"Duplicate headers: {cors.allow_headers}"
    print("Assertion passed - no duplicates")
except AssertionError as e:
    print(f"Assertion failed: {e}")