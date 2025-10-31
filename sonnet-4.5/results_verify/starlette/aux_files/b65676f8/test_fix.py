"""Test the proposed fix for the duplicate headers bug"""

# Simulate the SAFELISTED_HEADERS constant
SAFELISTED_HEADERS = {"Accept", "Accept-Language", "Content-Language", "Content-Type"}

def original_implementation(allow_headers):
    """Original buggy implementation"""
    # Line 58 equivalent
    allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
    # Line 67 equivalent
    final_headers = [h.lower() for h in allow_headers]
    return final_headers

def proposed_fix(allow_headers):
    """Proposed fixed implementation"""
    # Lowercase before merging to avoid case-insensitive duplicates
    allow_headers = sorted(set(h.lower() for h in SAFELISTED_HEADERS) | set(h.lower() for h in allow_headers))
    # No need to lowercase again
    final_headers = list(allow_headers)
    return final_headers

# Test cases
test_cases = [
    ["accept"],  # Lowercase version of safelisted header
    ["Accept"],  # Titlecase version of safelisted header
    ["accept", "x-custom-header"],  # Mix of safelisted and custom
    ["ACCEPT", "CONTENT-TYPE"],  # Uppercase versions
    ["x-custom-1", "x-custom-2"],  # Only custom headers
    [],  # Empty list
]

print("Testing original vs fixed implementation:\n")
for headers in test_cases:
    print(f"Input: {headers}")

    original = original_implementation(headers.copy())
    fixed = proposed_fix(headers.copy())

    original_dupes = len(original) != len(set(original))
    fixed_dupes = len(fixed) != len(set(fixed))

    print(f"  Original: {len(original)} headers, duplicates: {original_dupes}")
    if original_dupes:
        print(f"    Headers: {original}")

    print(f"  Fixed:    {len(fixed)} headers, duplicates: {fixed_dupes}")
    if fixed_dupes:
        print(f"    Headers: {fixed}")

    if original_dupes and not fixed_dupes:
        print("  ✓ Fix resolves the duplicate issue!")
    elif not original_dupes and not fixed_dupes:
        print("  ✓ No duplicates in either version")

    print()