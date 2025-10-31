import pandas.io.json._normalize as normalize

# Test cases to demonstrate the bug
test_cases = [
    ("00", "00"),  # Non-array string should be unchanged
    ('{"foo": "bar"}', '{"foo": "bar"}'),  # JSON object should be unchanged
    ("x]", "x]"),  # String ending with ] but not starting with [ should be unchanged
    ("[1, 2, 3]", "Line-delimited format"),  # JSON array should be converted
]

print("Testing convert_to_line_delimits function:")
print("=" * 50)

for input_str, expected_desc in test_cases:
    result = normalize.convert_to_line_delimits(input_str)

    if expected_desc == "Line-delimited format":
        # For arrays, we expect line-delimited output
        print(f"Input: {input_str!r}")
        print(f"Output: {result!r}")
        print(f"Expected: Line-delimited JSON format")
    else:
        # For non-arrays, output should equal input
        if result != input_str:
            print(f"BUG FOUND!")
            print(f"  Input:    {input_str!r}")
            print(f"  Output:   {result!r}")
            print(f"  Expected: {expected_desc!r}")
        else:
            print(f"OK: {input_str!r} -> {result!r}")
    print("-" * 50)