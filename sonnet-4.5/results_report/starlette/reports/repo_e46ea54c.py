from starlette.staticfiles import StaticFiles
from starlette.datastructures import Headers

# Create StaticFiles instance - using /tmp as it exists on all systems
static_files = StaticFiles(directory="/tmp", check_dir=False)

# Test 1: Weak ETag should match itself
response_headers = Headers({"etag": 'W/"123"'})
request_headers = Headers({"if-none-match": 'W/"123"'})
result = static_files.is_not_modified(response_headers, request_headers)
print(f"Test 1 - Weak ETag W/\"123\" matches itself: {result}")
print(f"  Expected: True, Got: {result}")

# Test 2: Strong ETag should match itself
response_headers2 = Headers({"etag": '"456"'})
request_headers2 = Headers({"if-none-match": '"456"'})
result2 = static_files.is_not_modified(response_headers2, request_headers2)
print(f"\nTest 2 - Strong ETag \"456\" matches itself: {result2}")
print(f"  Expected: True, Got: {result2}")

# Test 3: Multiple ETags with weak ETag
response_headers3 = Headers({"etag": 'W/"789"'})
request_headers3 = Headers({"if-none-match": '"123", W/"789", "456"'})
result3 = static_files.is_not_modified(response_headers3, request_headers3)
print(f"\nTest 3 - Weak ETag W/\"789\" in list: {result3}")
print(f"  Expected: True, Got: {result3}")

# Demonstrate the problem in the code
print("\n--- Debugging the issue ---")
etag = 'W/"123"'
if_none_match = 'W/"123"'
tags = [tag.strip(" W/") for tag in if_none_match.split(",")]
print(f"Response ETag: {etag}")
print(f"Request If-None-Match: {if_none_match}")
print(f"Stripped request tags: {tags}")
print(f"Is '{etag}' in {tags}? {etag in tags}")
print("\nThe bug: response ETag 'W/\"123\"' is never normalized, but request tags are.")
print("So we compare 'W/\"123\"' against ['\"123\"'], which fails!")