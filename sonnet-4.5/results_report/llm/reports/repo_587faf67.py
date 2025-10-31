from llm.utils import truncate_string

# Test case 1: max_length=1
print("Test case 1: max_length=1")
result = truncate_string("hello", max_length=1)
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: 1")
print(f"Violation: {len(result) > 1}")
print()

# Test case 2: max_length=0
print("Test case 2: max_length=0")
result = truncate_string("hello", max_length=0)
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: 0")
print(f"Violation: {len(result) > 0}")
print()

# Test case 3: max_length=2
print("Test case 3: max_length=2")
result = truncate_string("hello", max_length=2)
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: 2")
print(f"Violation: {len(result) > 2}")
print()

# Test minimal failing case: text="ab", max_length=1
print("Test minimal case: text='ab', max_length=1")
result = truncate_string("ab", max_length=1)
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: 1")
print(f"Violation: {len(result) > 1}")
print()

# Assertion test that will fail
print("Running assertion test...")
try:
    result = truncate_string("hello", max_length=1)
    assert len(result) <= 1, f"Length {len(result)} exceeds max_length of 1"
    print("Assertion passed")
except AssertionError as e:
    print(f"AssertionError: {e}")