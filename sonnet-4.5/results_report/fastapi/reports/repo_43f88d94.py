from starlette.middleware.cors import CORSMiddleware

# Create two middleware instances with the same header but different capitalization
middleware_upper = CORSMiddleware(None, allow_headers=['A'])
middleware_lower = CORSMiddleware(None, allow_headers=['a'])

print("With 'A':", middleware_upper.allow_headers)
print("With 'a':", middleware_lower.allow_headers)
print("Equal?", middleware_upper.allow_headers == middleware_lower.allow_headers)

# Additional test with another header
middleware_upper2 = CORSMiddleware(None, allow_headers=['X-Custom-Header'])
middleware_lower2 = CORSMiddleware(None, allow_headers=['x-custom-header'])

print("\nWith 'X-Custom-Header':", middleware_upper2.allow_headers)
print("With 'x-custom-header':", middleware_lower2.allow_headers)
print("Equal?", middleware_upper2.allow_headers == middleware_lower2.allow_headers)

# Test with multiple headers showing the sorting issue
middleware_multi_upper = CORSMiddleware(None, allow_headers=['B', 'A'])
middleware_multi_lower = CORSMiddleware(None, allow_headers=['b', 'a'])

print("\nWith ['B', 'A']:", middleware_multi_upper.allow_headers)
print("With ['b', 'a']:", middleware_multi_lower.allow_headers)
print("Equal?", middleware_multi_upper.allow_headers == middleware_multi_lower.allow_headers)