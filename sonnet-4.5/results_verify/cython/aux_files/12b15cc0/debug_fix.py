#!/usr/bin/env python3
"""Debug the fix more carefully"""

def trace_both_versions(s):
    """Trace both original and fixed versions side by side"""

    print(f"Input: '{s}', Characters: {list(s)}, Sorted: {sorted(s)}")
    print("=" * 60)

    # Original version with >=
    print("ORIGINAL VERSION (with >=):")
    char_list = sorted(s)
    i = 0
    n = len(char_list)
    result_orig = []
    while i < n:
        print(f"  Outer loop: i={i}, char='{char_list[i]}'")
        code1 = ord(char_list[i])
        code2 = code1 + 1
        i += 1
        print(f"    Initial: code1={code1}('{chr(code1)}'), code2={code2}('{chr(code2) if code2 < 128 else '?'}')")

        while i < n and code2 >= ord(char_list[i]):
            print(f"    Inner check: i={i}, char='{char_list[i]}', ord={ord(char_list[i])}, code2={code2}")
            print(f"      Condition: {code2} >= {ord(char_list[i])} = {code2 >= ord(char_list[i])}")
            code2 += 1
            i += 1
            print(f"      After: code2={code2}, i={i}")

        result_orig.append(code1)
        result_orig.append(code2)
        print(f"    Added range: [{code1}, {code2})")

    print(f"  Result: {result_orig}")

    print()
    print("FIXED VERSION (with >):")
    char_list = sorted(s)
    i = 0
    n = len(char_list)
    result_fixed = []
    while i < n:
        print(f"  Outer loop: i={i}, char='{char_list[i]}'")
        code1 = ord(char_list[i])
        code2 = code1 + 1
        i += 1
        print(f"    Initial: code1={code1}('{chr(code1)}'), code2={code2}('{chr(code2) if code2 < 128 else '?'}')")

        while i < n and code2 > ord(char_list[i]):
            print(f"    Inner check: i={i}, char='{char_list[i]}', ord={ord(char_list[i])}, code2={code2}")
            print(f"      Condition: {code2} > {ord(char_list[i])} = {code2 > ord(char_list[i])}")
            code2 += 1
            i += 1
            print(f"      After: code2={code2}, i={i}")

        result_fixed.append(code1)
        result_fixed.append(code2)
        print(f"    Added range: [{code1}, {code2})")

    print(f"  Result: {result_fixed}")

    return result_orig, result_fixed

# Test with '00'
print("TEST CASE: '00' (two identical characters)")
print("=" * 80)
orig, fixed = trace_both_versions('00')
print()

# Test with '01' (consecutive characters)
print("TEST CASE: '01' (consecutive characters)")
print("=" * 80)
orig2, fixed2 = trace_both_versions('01')
print()

# Test with 'abc' (multiple consecutive)
print("TEST CASE: 'abc' (multiple consecutive)")
print("=" * 80)
orig3, fixed3 = trace_both_versions('abc')