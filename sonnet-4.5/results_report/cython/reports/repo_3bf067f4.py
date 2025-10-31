from Cython.Plex.Regexps import chars_to_ranges

# Test with duplicate characters
s = '00'
ranges = chars_to_ranges(s)

covered = set()
for i in range(0, len(ranges), 2):
    code1, code2 = ranges[i], ranges[i+1]
    for code in range(code1, code2):
        covered.add(chr(code))

print(f"Input: {s!r}")
print(f"Ranges returned: {ranges}")
print(f"Expected coverage: {set(s)}")
print(f"Actual coverage: {covered}")
print(f"Extra characters incorrectly covered: {covered - set(s)}")

# Test with more duplicates
print("\n--- Test with more duplicates ---")
s2 = '0000000000'
ranges2 = chars_to_ranges(s2)

covered2 = set()
for i in range(0, len(ranges2), 2):
    code1, code2 = ranges2[i], ranges2[i+1]
    for code in range(code1, code2):
        covered2.add(chr(code))

print(f"Input: {s2!r}")
print(f"Ranges returned: {ranges2}")
print(f"Expected coverage: {set(s2)}")
print(f"Actual coverage: {covered2}")
print(f"Extra characters incorrectly covered: {covered2 - set(s2)}")

# Demonstrate impact on Any() and AnyBut()
print("\n--- Impact on Any() and AnyBut() ---")
from Cython.Plex import Lexicon, Scanner, Any, AnyBut, TEXT
from io import StringIO

# Any('00') should only match '0', but incorrectly matches '1' as well
print("Testing Any('00'):")
lexicon_any = Lexicon([(Any('00'), TEXT)])
scanner_any = Scanner(lexicon_any, StringIO('01'))
result = scanner_any.read()
if result:
    print(f"  Any('00') matched: {result[1]!r} (should only match '0')")

# AnyBut('00') should match '1', but raises UnrecognizedInput
print("\nTesting AnyBut('00'):")
try:
    lexicon_anybut = Lexicon([(AnyBut('00'), TEXT)])
    scanner_anybut = Scanner(lexicon_anybut, StringIO('1'))
    result = scanner_anybut.read()
    if result:
        print(f"  AnyBut('00') matched: {result[1]!r}")
except Exception as e:
    print(f"  AnyBut('00') raised error: {e.__class__.__name__}: {e}")
    print(f"  (This should NOT happen - '1' should be matched by AnyBut('00'))")