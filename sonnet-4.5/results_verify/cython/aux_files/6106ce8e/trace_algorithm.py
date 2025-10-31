#!/usr/bin/env python3
"""Trace through the algorithm to understand the bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

def chars_to_ranges_traced(s):
    """Traced version of the function to see step-by-step execution"""
    print(f"Input string: '{s}'")
    char_list = list(s)
    char_list.sort()
    print(f"Sorted char list: {char_list}")

    i = 0
    n = len(char_list)
    result = []

    while i < n:
        print(f"\n--- Starting new range at i={i} ---")
        code1 = ord(char_list[i])
        print(f"code1 = ord('{char_list[i]}') = {code1}")
        code2 = code1 + 1
        print(f"code2 = {code2}")
        i += 1
        print(f"i incremented to {i}")

        while i < n and code2 >= ord(char_list[i]):
            print(f"  Checking: i={i}, code2={code2}, ord('{char_list[i]}')={ord(char_list[i])}")
            print(f"  Condition: {code2} >= {ord(char_list[i])} is {code2 >= ord(char_list[i])}")
            code2 += 1
            print(f"  code2 incremented to {code2}")
            i += 1
            print(f"  i incremented to {i}")

        result.append(code1)
        result.append(code2)
        print(f"Added range [{code1}, {code2}) which covers: ", end="")
        chars = [chr(c) for c in range(code1, code2)]
        print(chars)

    return result

# Test with '00'
print("=" * 60)
print("Test with '00':")
result = chars_to_ranges_traced('00')
print(f"\nFinal result: {result}")
print(f"This represents range [48, 50) which includes characters: {[chr(c) for c in range(48, 50)]}")

print("\n" + "=" * 60)
print("Test with 'aaa':")
result = chars_to_ranges_traced('aaa')
print(f"\nFinal result: {result}")
print(f"This represents range [97, 100) which includes characters: {[chr(c) for c in range(97, 100)]}")

print("\n" + "=" * 60)
print("\nThe bug explanation:")
print("When code2 >= ord(char_list[i]), the condition is TRUE even when they're equal.")
print("This means duplicate characters cause code2 to increment, extending the range.")
print("The fix should be: while i < n and code2 > ord(char_list[i])")
print("This way, duplicate characters (where code2 == ord(char_list[i])+1) won't extend the range.")