from Cython.Utils import build_hex_version

# Test the specific failing case mentioned in the report
version = '0.0rc'
print(f"Testing version: '{version}'")
try:
    result = build_hex_version(version)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Also test other similar cases
test_cases = ['0.0a', '0.0b', '1.0rc', '2.3.4a', '2.3.4rc']
for version in test_cases:
    print(f"\nTesting version: '{version}'")
    try:
        result = build_hex_version(version)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")