from starlette.staticfiles import StaticFiles
from starlette.datastructures import Headers

print("=== Testing ETag comparison bug ===")

# Create a StaticFiles instance
static_files = StaticFiles(directory="/tmp", check_dir=False)

# Test 1: Weak ETag should match itself
print("\nTest 1: Weak ETag comparison")
response_headers = Headers({"etag": 'W/"123"'})
request_headers = Headers({"if-none-match": 'W/"123"'})

result = static_files.is_not_modified(response_headers, request_headers)
print(f"Response ETag: {response_headers['etag']}")
print(f"Request If-None-Match: {request_headers['if-none-match']}")
print(f"Result (should be True): {result}")

# Test 2: Strong ETag should match itself
print("\nTest 2: Strong ETag comparison")
response_headers2 = Headers({"etag": '"456"'})
request_headers2 = Headers({"if-none-match": '"456"'})

result2 = static_files.is_not_modified(response_headers2, request_headers2)
print(f"Response ETag: {response_headers2['etag']}")
print(f"Request If-None-Match: {request_headers2['if-none-match']}")
print(f"Result (should be True): {result2}")

# Test 3: Multiple ETags in If-None-Match
print("\nTest 3: Multiple ETags in If-None-Match")
response_headers3 = Headers({"etag": 'W/"789"'})
request_headers3 = Headers({"if-none-match": '"111", W/"789", "222"'})

result3 = static_files.is_not_modified(response_headers3, request_headers3)
print(f"Response ETag: {response_headers3['etag']}")
print(f"Request If-None-Match: {request_headers3['if-none-match']}")
print(f"Result (should be True): {result3}")

# Test 4: Testing the strip bug
print("\nTest 4: Testing edge case with strip()")
# This demonstrates the strip character issue
test_string = 'W/"abc"//WW'
stripped = test_string.strip(" W/")
print(f"Original: {test_string}")
print(f"After .strip(' W/'): {stripped}")
print("Note: strip() removes individual characters, not the string 'W/'")