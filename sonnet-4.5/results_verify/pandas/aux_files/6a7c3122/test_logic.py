from pandas.io.json._normalize import convert_to_line_delimits

# Let's understand the condition:
# if not s[0] == "[" and s[-1] == "]":
#     return s

print("Understanding the boolean logic:")
print("The condition is: if not s[0] == '[' and s[-1] == ']':")
print("Due to operator precedence, this is: if (not s[0] == '[') and (s[-1] == ']'):")
print("Which is equivalent to: if (s[0] != '[') and (s[-1] == ']'):")
print()

# This means it returns early when:
# - The string does NOT start with '['
# - AND the string ends with ']'

test_cases = [
    ('[1, 2, 3]', 'Valid JSON array'),
    ('{"a": 1}', 'Valid JSON object'),
    ('{"a": 1}]', 'Malformed: object start, array end'),
    ('[1, 2, 3}', 'Malformed: array start, object end'),
]

for input_str, description in test_cases:
    starts_with_bracket = input_str[0] == '['
    ends_with_bracket = input_str[-1] == ']'

    # Current logic
    current_condition = (not input_str[0] == '[') and (input_str[-1] == ']')

    # Expected logic (should only process arrays)
    expected_should_process = starts_with_bracket and ends_with_bracket

    result = convert_to_line_delimits(input_str)
    was_processed = result != input_str

    print(f"Input: {input_str:<15} ({description})")
    print(f"  Starts with '[': {starts_with_bracket}")
    print(f"  Ends with ']': {ends_with_bracket}")
    print(f"  Current condition (returns early): {current_condition}")
    print(f"  Was processed: {was_processed}")
    print(f"  Should process (if valid array): {expected_should_process}")
    print(f"  Bug: {was_processed != expected_should_process}")
    print()