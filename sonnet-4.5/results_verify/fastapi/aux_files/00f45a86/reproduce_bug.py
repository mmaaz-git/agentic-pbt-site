from fastapi.security.oauth2 import SecurityScopes

# Test case from the bug report
scopes_list = ['\x85']
security_scopes = SecurityScopes(scopes=scopes_list)

print(f"Input scopes: {scopes_list}")
print(f"scope_str: {security_scopes.scope_str!r}")
print(f"Reconstructed: {security_scopes.scope_str.split()}")

try:
    assert security_scopes.scope_str.split() == scopes_list
    print("Assertion passed - no bug")
except AssertionError:
    print("AssertionError - bug confirmed!")

# Let's also test some other Unicode whitespace characters
test_cases = [
    '\x85',  # NEL (Next Line)
    '\xa0',  # NBSP (Non-breaking space)
    '\u2000',  # En Quad
    '\u2028',  # Line separator
    '\u2029',  # Paragraph separator
    'valid_scope',  # Normal case
]

print("\nTesting various characters:")
for char in test_cases:
    ss = SecurityScopes(scopes=[char])
    reconstructed = ss.scope_str.split()
    match = reconstructed == [char]
    print(f"  {char!r}: '{ss.scope_str}' -> {reconstructed} {'✓' if match else '✗'}")