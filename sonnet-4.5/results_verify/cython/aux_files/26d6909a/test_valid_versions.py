from Cython.Utils import build_hex_version

# Test valid versions that work
test_cases = [
    '1.0',
    '1.2.3',
    '1.2.3a1',
    '1.2.3b2',
    '1.2.3rc3',
    '4.3a1',  # Example from docstring
]

for version in test_cases:
    try:
        result = build_hex_version(version)
        print(f"'{version}' → {result}")
    except Exception as e:
        print(f"'{version}' → ERROR: {e}")