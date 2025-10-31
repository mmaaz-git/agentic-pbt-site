from starlette.middleware.cors import CORSMiddleware

async def dummy_app(scope, receive, send):
    pass

# Test case 1: Duplicates bug
middleware = CORSMiddleware(app=dummy_app, allow_headers=['Q', 'q'])
print("Test 1 - Duplicates Bug:")
print("Input headers: ['Q', 'q']")
print("Result headers:", middleware.allow_headers)
print("Has duplicates:", len(middleware.allow_headers) != len(set(middleware.allow_headers)))
print("Expected: Headers should be deduplicated since 'Q' and 'q' are the same header (case-insensitive)")
print()

# Test case 2: Sorting bug
middleware2 = CORSMiddleware(app=dummy_app, allow_headers=['['])
print("Test 2 - Sorting Bug:")
print("Input headers: ['[']")
print("Result headers:", middleware2.allow_headers)
print("Is properly sorted:", middleware2.allow_headers == sorted(middleware2.allow_headers))
print("Expected: Headers should be in alphabetical order")