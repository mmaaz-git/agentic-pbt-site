"""Test to reproduce the OAuth2 scope parsing bug"""

# First, let's test the hypothesis test case
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1).filter(lambda x: " " not in x), min_size=1))
def test_oauth2_scope_spec_compliance(scopes_list):
    scope_string = " ".join(scopes_list)

    parsed_scopes_with_split = scope_string.split()
    parsed_scopes_with_space_split = scope_string.split(" ")

    assert parsed_scopes_with_split == parsed_scopes_with_space_split, \
        f"Failed with scopes_list={scopes_list!r}, split()={parsed_scopes_with_split}, split(' ')={parsed_scopes_with_space_split}"

# Test with the reported failing input
print("Testing with failing input: scopes_list = ['\\r']")
scopes_list = ['\r']
scope_string = " ".join(scopes_list)
parsed_with_split = scope_string.split()
parsed_with_space_split = scope_string.split(" ")
print(f"scope_string: {scope_string!r}")
print(f"split(): {parsed_with_split}")
print(f"split(' '): {parsed_with_space_split}")
print(f"Are they equal? {parsed_with_split == parsed_with_space_split}\n")

# Now test the malicious scope example
print("Testing malicious scope example:")
malicious_scope = "read\nwrite"

parsed = malicious_scope.split()
print(f"Parsed scopes with split(): {parsed}")

expected_per_spec = malicious_scope.split(" ")
print(f"Expected per OAuth2 spec with split(' '): {expected_per_spec}")

print(f"\nAssertion: parsed == ['read', 'write'] -> {parsed == ['read', 'write']}")
print(f"Assertion: expected_per_spec == ['read\\nwrite'] -> {expected_per_spec == ['read\nwrite']}")

# Additional test cases to understand the behavior
print("\n--- Additional test cases ---")

test_cases = [
    "read write",           # Normal case with space
    "read\twrite",          # Tab character
    "read\nwrite",          # Newline character
    "read\rwrite",          # Carriage return
    "read  write",          # Multiple spaces
    "read\n\nwrite",        # Multiple newlines
    " read write ",         # Leading/trailing spaces
    "read",                 # Single scope
    "",                     # Empty string
]

for test_case in test_cases:
    split_result = test_case.split()
    space_split_result = test_case.split(" ")
    print(f"Input: {test_case!r}")
    print(f"  split(): {split_result}")
    print(f"  split(' '): {space_split_result}")
    print(f"  Equal? {split_result == space_split_result}")
    print()