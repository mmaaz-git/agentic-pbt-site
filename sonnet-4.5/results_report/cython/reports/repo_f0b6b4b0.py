from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

# Test cases that should all return False according to the docstring
test_cases = [".0", ".1", ".123", ".999", ".0000"]

print("Testing is_valid_tag with strings that should be rejected:")
print("=" * 60)

for test_string in test_cases:
    regular_result = is_valid_tag(test_string)
    encoded_result = is_valid_tag(EncodedString(test_string))

    print(f"Input: '{test_string}'")
    print(f"  Regular string: is_valid_tag('{test_string}') = {regular_result}")
    print(f"  EncodedString:  is_valid_tag(EncodedString('{test_string}')) = {encoded_result}")

    if regular_result != encoded_result:
        print(f"  ⚠️  INCONSISTENCY: Regular string returns {regular_result}, EncodedString returns {encoded_result}")
    print()

print("=" * 60)
print("Testing edge cases that should return True:")
print("=" * 60)

valid_cases = [".a", ".1a", "0.", "normal_name", "_private", ""]
for test_string in valid_cases:
    regular_result = is_valid_tag(test_string)
    encoded_result = is_valid_tag(EncodedString(test_string))

    print(f"Input: '{test_string}'")
    print(f"  Regular string: {regular_result}")
    print(f"  EncodedString:  {encoded_result}")
    if regular_result != encoded_result:
        print(f"  ⚠️  INCONSISTENCY!")
    print()