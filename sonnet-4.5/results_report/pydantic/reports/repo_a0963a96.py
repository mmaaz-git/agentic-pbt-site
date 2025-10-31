from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated

class Model(BaseModel):
    value: Annotated[str, validate_as(str).str_strip()]

# Test with Unicode Unit Separator (U+001F)
m = Model(value='\x1f')
print(f"Result: {m.value!r}")
print(f"Expected: {'\x1f'.strip()!r}")
print(f"Are they equal? {m.value == '\x1f'.strip()}")

# Additional test with various Unicode whitespace
test_strings = [
    '\x1f',  # Unicode Unit Separator
    '\x1c',  # File Separator
    '\x1d',  # Group Separator
    '\x1e',  # Record Separator
    '\x0b',  # Vertical Tab
    '\x0c',  # Form Feed
    '\x85',  # Next Line
    '\xa0',  # Non-breaking space
    '\u2000',  # En Quad
    '\u2001',  # Em Quad
    '\u2002',  # En Space
    '\u2003',  # Em Space
    '\u2004',  # Three-Per-Em Space
    '\u2005',  # Four-Per-Em Space
    '\u2006',  # Six-Per-Em Space
    '\u2007',  # Figure Space
    '\u2008',  # Punctuation Space
    '\u2009',  # Thin Space
    '\u200a',  # Hair Space
    '\u202f',  # Narrow No-Break Space
    '\u205f',  # Medium Mathematical Space
    '\u3000',  # Ideographic Space
]

print("\n--- Testing various Unicode whitespace characters ---")
for test in test_strings:
    model = Model(value=test)
    expected = test.strip()
    print(f"Input: {test!r} (U+{ord(test):04X})")
    print(f"  pydantic result: {model.value!r}")
    print(f"  Python strip(): {expected!r}")
    print(f"  Match: {model.value == expected}")
    if model.value != expected:
        print(f"  ❌ MISMATCH!")
    print()