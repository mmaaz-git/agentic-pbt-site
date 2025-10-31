from Cython.Plex.Regexps import chars_to_ranges

# Test with duplicate characters
test_cases = ['00', 'aaa', 'aabbcc']

for s in test_cases:
    print(f"\n=== Testing input: '{s}' ===")
    ranges = chars_to_ranges(s)
    print(f"Returned ranges: {ranges}")

    # Decode what characters are covered by these ranges
    covered_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i + 1]
        print(f"Range [{code1}, {code2}): ", end="")
        chars_in_range = []
        for code in range(code1, code2):
            char = chr(code)
            covered_chars.add(char)
            chars_in_range.append(f"'{char}'")
        print(", ".join(chars_in_range))

    original_chars = set(s)
    print(f"Original characters: {original_chars}")
    print(f"Covered characters: {covered_chars}")

    if covered_chars == original_chars:
        print("✓ CORRECT: Ranges cover exactly the input characters")
    else:
        extra = covered_chars - original_chars
        missing = original_chars - covered_chars
        if extra:
            print(f"✗ BUG: Ranges include extra characters: {extra}")
        if missing:
            print(f"✗ BUG: Ranges missing characters: {missing}")