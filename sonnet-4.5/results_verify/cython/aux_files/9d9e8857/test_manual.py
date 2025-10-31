from Cython.Compiler.PyrexTypes import cap_length

# Test the exact case from the bug report
result = cap_length('0', max_len=0)
print(f"Result: {repr(result)}")
print(f"Result length: {len(result)}")
print(f"Expected max length: 0")

# Test other small max_len values
print("\nTesting other small max_len values:")
for max_len in range(0, 15):
    for test_str in ['a', 'ab', 'abc', 'abcd', '0' * 20]:
        result = cap_length(test_str, max_len)
        if len(result) > max_len:
            print(f"FAIL: cap_length({repr(test_str)}, {max_len}) = {repr(result)} (len={len(result)})")