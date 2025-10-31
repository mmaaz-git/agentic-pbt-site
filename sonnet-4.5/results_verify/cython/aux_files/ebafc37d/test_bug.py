#!/usr/bin/env python3
"""Test script to reproduce the reported bug in parse_list"""

from Cython.Build.Dependencies import parse_list
import traceback

def test_case(test_input, description):
    """Test a single input and report results"""
    print(f"\nTesting: {description}")
    print(f"Input: {repr(test_input)}")
    try:
        result = parse_list(test_input)
        print(f"Result: {result}")
        return True
    except KeyError as e:
        print(f"KeyError: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Other Exception ({type(e).__name__}): {e}")
        traceback.print_exc()
        return False

print("=" * 60)
print("REPRODUCING BUG REPORT TEST CASES")
print("=" * 60)

# Main failing case from report
test_case('[""]', 'Empty string in brackets')

# Additional cases from report
test_case("'", 'Single quote alone')
test_case('["\\"]', 'Escaped quote in brackets')
test_case('[a, "", b]', 'Empty string with other elements')
test_case('"', 'Double quote alone')

# Some additional edge cases to understand behavior
print("\n" + "=" * 60)
print("ADDITIONAL TEST CASES")
print("=" * 60)

test_case("['']", 'Empty single-quoted string in brackets')
test_case('""', 'Empty double-quoted string without brackets')
test_case("''", 'Empty single-quoted string without brackets')
test_case('["a"]', 'Non-empty quoted string in brackets')
test_case('"a"', 'Non-empty quoted string without brackets')

# Test cases from the docstring to verify they still work
print("\n" + "=" * 60)
print("DOCSTRING EXAMPLES (should all pass)")
print("=" * 60)

test_case("", "Empty input")
test_case("a", "Single element")
test_case("a b c", "Space-separated")
test_case("[a, b, c]", "Bracket notation")
test_case('a " " b', 'Quoted space')
test_case('[a, ",a", "a,", ",", ]', 'Quoted commas')

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)