from fastapi.security.utils import get_authorization_scheme_param

# Test how get_authorization_scheme_param handles different inputs
test_cases = [
    None,
    "",
    "Bearer",
    "Bearer ",
    "Bearer  ",
    "Bearer token123",
    "Basic dXNlcjpwYXNz",
    "InvalidScheme",
]

print("Testing get_authorization_scheme_param function:")
print("-" * 50)
for test in test_cases:
    scheme, param = get_authorization_scheme_param(test)
    print(f"Input: {repr(test)}")
    print(f"  -> scheme: {repr(scheme)}, param: {repr(param)}")
    print()