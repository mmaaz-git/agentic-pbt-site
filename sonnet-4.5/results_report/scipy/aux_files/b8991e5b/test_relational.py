#!/usr/bin/env python3
"""Test demonstrating how empty strings occur naturally when parsing relational data."""

# Simulate what happens in RelationalAttribute.parse_data()
data_examples = [
    "value1,value2\nvalue3,value4\n",  # Ends with newline (common)
    "value1,value2\n\nvalue3,value4",   # Double newline
    "\nvalue1,value2",                   # Starts with newline
    "value1,value2\n",                   # Single line ending with newline
]

print("Demonstrating how split() produces empty strings:\n")

for i, data_str in enumerate(data_examples, 1):
    print(f"Example {i}: {repr(data_str)}")
    lines = data_str.split("\n")
    print(f"  Split result: {lines}")
    empty_count = sum(1 for line in lines if line == '')
    if empty_count > 0:
        print(f"  ⚠️  Contains {empty_count} empty string(s)!")
    print()

print("-" * 60)
print("\nThis shows that empty strings naturally occur when splitting")
print("relational attribute data by newlines, which would cause")
print("split_data_line() to crash with IndexError.")