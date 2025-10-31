from Cython.Utils import build_hex_version

# Test various cases
test_cases = [
    '0.0rc',     # Should fail
    '0.0a',      # Should fail
    '0.0b',      # Should fail
    '1.0rc',     # Should fail
    '2.3.4a',    # Should fail
    '1.0rc1',    # Should work
    '1.0a2',     # Should work
    '1.0b3',     # Should work
]

for version in test_cases:
    try:
        result = build_hex_version(version)
        print(f"✓ '{version}' → {result}")
    except ValueError as e:
        print(f"✗ '{version}' → ValueError: {e}")