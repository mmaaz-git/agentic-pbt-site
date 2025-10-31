from pydantic import BaseModel, Field

# Test which whitespace characters are stripped by pydantic vs Python
class TestStripModel(BaseModel):
    field: str = Field(strip_whitespace=True)

# Test various whitespace characters
test_chars = [
    (0x09, 'TAB'),
    (0x0A, 'LF'),
    (0x0B, 'VT'),
    (0x0C, 'FF'),
    (0x0D, 'CR'),
    (0x20, 'SPACE'),
    (0x1C, 'FS'),
    (0x1D, 'GS'),
    (0x1E, 'RS'),
    (0x1F, 'US'),
    (0x85, 'NEL'),
    (0xA0, 'NBSP'),
    (0x2000, 'EN QUAD'),
]

print("Comparing strip behavior:")
print("Code | Name      | Python strip | Pydantic strip")
print("-----|-----------|--------------|---------------")
for code, name in test_chars:
    char = chr(code)
    test_str = f'{char}x{char}'  # Surround x with the character

    # Python's str.strip()
    python_result = test_str.strip()
    python_stripped = python_result == 'x'

    # Pydantic's strip_whitespace
    model = TestStripModel(field=test_str)
    pydantic_result = model.field
    pydantic_stripped = pydantic_result == 'x'

    match = '✓' if python_stripped == pydantic_stripped else '✗'
    print(f"0x{code:02X} | {name:9} | {str(python_stripped):12} | {str(pydantic_stripped):12} {match}")

# Test specific examples with common whitespace
print("\nSpecific examples:")
for test in [' x ', '\tx\t', '\nx\n', '  x  ']:
    model = TestStripModel(field=test)
    print(f"Input: {test!r} -> Python: {test.strip()!r}, Pydantic: {model.field!r}")