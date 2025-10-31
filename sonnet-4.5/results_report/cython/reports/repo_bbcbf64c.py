from Cython.Utils import build_hex_version

# Test case that crashes
version = '0.0rc'
print(f"Testing build_hex_version('{version}')")
result = build_hex_version(version)
print(f"Result: {result}")