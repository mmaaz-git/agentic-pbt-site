#!/usr/bin/env python3
"""Test to reproduce the FastMachine.chars_to_ranges duplicate character bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Plex.Machines import FastMachine

print("=" * 60)
print("Testing FastMachine.chars_to_ranges with duplicate characters")
print("=" * 60)

fm = FastMachine()

# Test case from bug report
print("\n1. Test with ['0', '0'] (duplicate characters)")
char_list1 = ['0', '0']
result1 = fm.chars_to_ranges(char_list1)
print(f"Input: {char_list1}")
print(f"Output: {result1}")
print(f"Unique characters in input: {set(char_list1)} = {sorted(set(char_list1))}")
print(f"Number of ranges: {len(result1)}")
if len(result1) == 1:
    print("✓ Produces single range as expected")
else:
    print("✗ Produces multiple ranges - potential bug!")

# Additional test cases
print("\n2. Test with ['a', 'a', 'a'] (triple duplicate)")
char_list2 = ['a', 'a', 'a']
result2 = fm.chars_to_ranges(char_list2)
print(f"Input: {char_list2}")
print(f"Output: {result2}")
print(f"Unique characters: {set(char_list2)}")
print(f"Number of ranges: {len(result2)}")

print("\n3. Test with ['a', 'b', 'b', 'c'] (consecutive with duplicate)")
char_list3 = ['a', 'b', 'b', 'c']
result3 = fm.chars_to_ranges(char_list3)
print(f"Input: {char_list3}")
print(f"Output: {result3}")
print(f"Expected for unique: {fm.chars_to_ranges(['a', 'b', 'c'])}")

print("\n4. Test with ['z', 'a', 'z', 'a'] (non-consecutive duplicates)")
char_list4 = ['z', 'a', 'z', 'a']
result4 = fm.chars_to_ranges(char_list4)
print(f"Input: {char_list4}")
print(f"Output: {result4}")
print(f"After sorting: {sorted(char_list4)}")

# Now test the property from the bug report
print("\n" + "=" * 60)
print("Testing the property from bug report:")
print("=" * 60)

from hypothesis import given, strategies as st

@given(st.lists(st.characters(), min_size=1, max_size=50))
def test_chars_to_ranges_consecutive_merging(char_list):
    fm = FastMachine()
    ranges = fm.chars_to_ranges(char_list)

    for i in range(len(ranges) - 1):
        c1_end = ord(ranges[i][1])
        c2_start = ord(ranges[i + 1][0])
        assert c1_end + 1 < c2_start, f"Ranges should not be adjacent or overlapping: {ranges[i]} and {ranges[i+1]}"

print("Running property-based test...")
try:
    test_chars_to_ranges_consecutive_merging()
    print("✗ Property test passed - no bug found!")
except AssertionError as e:
    print(f"✓ Property test failed as claimed: {e}")
except Exception as e:
    print(f"Property test failed with: {e}")
    import traceback
    traceback.print_exc()

# Let's manually trace through the algorithm with ['0', '0']
print("\n" + "=" * 60)
print("Manual trace of algorithm with ['0', '0']:")
print("=" * 60)

char_list = ['0', '0']
print(f"Input: {char_list}")
char_list_sorted = sorted(char_list)
print(f"After sorting: {char_list_sorted}")

print("\nStep-by-step execution:")
i = 0
n = len(char_list_sorted)
result = []
step = 1

while i < n:
    print(f"  Step {step}: i={i}, char='{char_list_sorted[i]}'")
    c1 = ord(char_list_sorted[i])
    c2 = c1
    print(f"    c1={c1} ('{chr(c1)}'), c2={c2} ('{chr(c2)}')")
    i += 1

    print(f"    Inner loop: checking if i < n and next char is consecutive")
    while i < n and ord(char_list_sorted[i]) == c2 + 1:
        print(f"      i={i}, char='{char_list_sorted[i]}', ord={ord(char_list_sorted[i])}, c2+1={c2+1}")
        i += 1
        c2 += 1
        print(f"      Updated: i={i}, c2={c2}")

    if i < n:
        print(f"    Loop stopped: i={i}, next char='{char_list_sorted[i]}', ord={ord(char_list_sorted[i])}, c2+1={c2+1}")
    else:
        print(f"    Loop stopped: reached end of list")

    range_tuple = (chr(c1), chr(c2))
    result.append(range_tuple)
    print(f"    Added range: {range_tuple}")
    step += 1

print(f"\nFinal result: {tuple(result)}")
print(f"Number of ranges: {len(result)}")

print("\n" + "=" * 60)
print("Analysis:")
print("=" * 60)
print("The issue is that when we have duplicate '0' characters:")
print("1. First iteration processes index 0, char '0'")
print("2. Inner loop checks if char at index 1 ('0') == c2 + 1 (48 + 1 = 49)")
print("3. Since ord('0') = 48 != 49, the loop doesn't continue")
print("4. First range ('0', '0') is added, i is now 1")
print("5. Second iteration processes index 1, another '0'")
print("6. Creates another range ('0', '0')")
print("\nResult: TWO identical ranges [('0', '0'), ('0', '0')] instead of one!")