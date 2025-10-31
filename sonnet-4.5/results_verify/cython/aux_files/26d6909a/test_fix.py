import re

def build_hex_version_fixed(version_string):
    """
    Parse and translate public version identifier like '4.3a1' into the readable hex representation '0x040300A1' (like PY_VERSION_HEX).

    SEE: https://peps.python.org/pep-0440/#public-version-identifiers
    """
    # Parse '4.12a1' into [4, 12, 0, 0xA01]
    # And ignore .dev, .pre and .post segments
    digits = []
    release_status = 0xF0
    for segment in re.split(r'(\D+)', version_string):
        if segment in ('a', 'b', 'rc'):
            release_status = {'a': 0xA0, 'b': 0xB0, 'rc': 0xC0}[segment]
            digits = (digits + [0, 0])[:3]  # 1.2a1 -> 1.2.0a1
        elif segment in ('.dev', '.pre', '.post'):
            break  # break since those are the last segments
        elif segment and segment != '.':  # FIX: Added check for empty string
            digits.append(int(segment))

    digits = (digits + [0] * 3)[:4]
    digits[3] += release_status

    # Then, build a single hex value, two hex digits per version part.
    hexversion = 0
    for digit in digits:
        hexversion = (hexversion << 8) + digit

    return '0x%08X' % hexversion

# Test the fixed version
test_cases = [
    '1.0',
    '1.2.3',
    '1.2.3a1',
    '1.0a',     # Should work now
    '1.0b',     # Should work now
    '1.0rc',    # Should work now
    '0.0rc',    # The original failing case
    '2.3.4a',   # Should work now
    '2.3.4rc',  # Should work now
]

print("Testing fixed version:")
for version in test_cases:
    try:
        result = build_hex_version_fixed(version)
        print(f"'{version}' → {result}")
    except Exception as e:
        print(f"'{version}' → ERROR: {e}")