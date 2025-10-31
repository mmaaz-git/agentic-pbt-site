import llm.utils

# Test case that demonstrates the bug
text = "Hello, World!"
max_length = 1

print(f"Input text: '{text}'")
print(f"Input text length: {len(text)}")
print(f"max_length: {max_length}")
print()

result = llm.utils.truncate_string(text, max_length=max_length)

print(f"Output: '{result}'")
print(f"Output length: {len(result)}")
print()
print(f"CONSTRAINT VIOLATION: len(result) = {len(result)} > max_length = {max_length}")
print(f"The function was asked to limit output to {max_length} character(s) but returned {len(result)} characters")