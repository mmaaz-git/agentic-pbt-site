from Cython.Compiler.PyrexTypes import cap_length

# Test with the minimal failing case from the report
result = cap_length('0', max_len=0)
print(f"Result: {repr(result)}")
print(f"Result length: {len(result)}")
print(f"Expected max length: 0")
print()

# Test with a few more cases to show the pattern
for max_len in [0, 5, 10, 12]:
    test_string = 'a' * 20  # A string longer than max_len
    result = cap_length(test_string, max_len=max_len)
    print(f"cap_length('{test_string[:10]}...', max_len={max_len})")
    print(f"  Result: {repr(result)}")
    print(f"  Result length: {len(result)} (expected <= {max_len})")
    if len(result) > max_len:
        print(f"  VIOLATION: Result exceeds max_len by {len(result) - max_len} characters")
    print()