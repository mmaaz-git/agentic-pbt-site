from dask.utils import key_split

# Test the main failing case
result = key_split('task-feedback-1')
print(f"Result: {repr(result)}")
print(f"Expected: 'task-feedback'")
print()

# Test if the bug occurs
try:
    assert result == 'task-feedback'
    print("✓ Test passed: 'feedback' was preserved")
except AssertionError:
    print("✗ Bug confirmed: 'feedback' was incorrectly stripped!")

print("\n--- Additional test cases ---")

# Test other affected words
test_cases = [
    ('process-feedback-0', 'process-feedback'),
    ('data-faceache-1', 'data-faceache'),
    ('task-beefcafe-2', 'task-beefcafe'),
    ('hello-world-1', 'hello-world'),  # This should work as shown in docstring
    ('x-abcdefab-1', 'x'),  # This is expected to strip per docstring
]

for input_str, expected in test_cases:
    result = key_split(input_str)
    status = "✓" if result == expected else "✗"
    print(f"{status} key_split('{input_str}') -> '{result}' (expected: '{expected}')")