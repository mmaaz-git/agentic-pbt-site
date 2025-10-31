#!/usr/bin/env python3
"""Test script to reproduce the reported bug in cap_length function"""

from hypothesis import given, strategies as st, settings
from Cython.Compiler import PyrexTypes

# First, let's test the specific failing case mentioned in the bug report
print("Testing specific failing case: s='00000000000', max_len=10")
result = PyrexTypes.cap_length('00000000000', 10)
print(f"Result: {result!r}")
print(f"Length: {len(result)}")
print(f"Expected: <= 10")
print(f"Actual: {len(result)}")
print(f"Test {'PASSED' if len(result) <= 10 else 'FAILED'}: Length is {len(result)}, max_len was 10")
print()

# Let's test a few more edge cases
test_cases = [
    ('a' * 20, 10),  # String longer than max_len=10
    ('b' * 20, 15),  # String longer than max_len=15
    ('c' * 20, 17),  # String longer than max_len=17 (boundary case)
    ('d' * 20, 20),  # String equal to max_len=20
    ('e' * 30, 25),  # String longer than max_len=25
    ('f' * 5, 10),   # String shorter than max_len
    ('g' * 100, 5),  # Very long string with very small max_len
    ('h' * 100, 12), # Very long string with max_len=12
    ('i' * 100, 13), # Very long string with max_len=13 (boundary)
    ('j' * 100, 16), # Very long string with max_len=16
]

print("Testing additional edge cases:")
for s, max_len in test_cases:
    result = PyrexTypes.cap_length(s, max_len)
    passed = len(result) <= max_len
    print(f"Input: s='{s[:10]}...' (len={len(s)}), max_len={max_len}")
    print(f"  Result: '{result}' (len={len(result)})")
    print(f"  Test {'PASSED' if passed else 'FAILED'}: {'✓' if passed else '✗'} (expected len <= {max_len}, got {len(result)})")
    print()

# Now run the property-based test
print("\nRunning property-based test from bug report:")
failure_count = 0
test_count = 0

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
       st.integers(min_value=10, max_value=200))
@settings(max_examples=1000)
def test_cap_length_respects_max_len(s, max_len):
    global failure_count, test_count
    test_count += 1
    result = PyrexTypes.cap_length(s, max_len)
    if len(result) > max_len:
        failure_count += 1
        if failure_count <= 5:  # Only print first 5 failures
            print(f"  FAILURE #{failure_count}: s='{s[:20]}{'...' if len(s) > 20 else ''}' (len={len(s)}), max_len={max_len}")
            print(f"    Result: '{result}' (len={len(result)})")
    assert len(result) <= max_len

try:
    test_cap_length_respects_max_len()
    print(f"\nProperty test completed: {test_count} tests, {failure_count} failures")
except AssertionError as e:
    print(f"\nProperty test failed after {test_count} tests with {failure_count} total failures")
    print("AssertionError caught - test failed as expected per bug report")

# Let's understand the implementation logic
print("\n" + "="*60)
print("Understanding the implementation:")
print("="*60)
print("\nCurrent implementation logic:")
print("  if len(s) <= max_len: return s")
print("  hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]")
print("  return '%s__%s__etc' % (hash_prefix, s[:max_len-17])")
print("\nFormat: {6-char-hash}__{truncated}__etc")
print("Fixed chars: 6 (hash) + 2 (__) + 5 (__etc) = 13 total fixed chars")
print("\nProblem: s[:max_len-17] when max_len < 17 results in negative slice")
print("  e.g., max_len=10: s[:10-17] = s[:-7] = 'all but last 7 chars'")
print("  This takes MORE of the string, not less!")

# Demonstrate the negative slice issue
print("\nDemonstrating negative slice behavior:")
test_string = "0123456789ABCDEF"
for max_len in [5, 10, 15, 17, 20]:
    slice_end = max_len - 17
    sliced = test_string[:slice_end]
    print(f"  max_len={max_len:2d}: s[:{slice_end:3d}] = '{sliced}' (len={len(sliced)})")