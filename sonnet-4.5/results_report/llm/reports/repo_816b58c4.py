from llm.utils import truncate_string

# Test case that crashes - max_length=1
result = truncate_string("hello world", 1)
print(f"Result: '{result}'")
print(f"Length: {len(result)}")
print(f"Expected max: 1")
print()

# Additional failing cases
print("Additional failing cases:")
test_cases = [
    ("test", 1),
    ("test", 2),
    ("example", 1),
    ("example", 2),
    ("a", 1),
    ("ab", 2),
]

for text, max_len in test_cases:
    result = truncate_string(text, max_len)
    print(f"truncate_string('{text}', {max_len}) = '{result}' (length={len(result)})")