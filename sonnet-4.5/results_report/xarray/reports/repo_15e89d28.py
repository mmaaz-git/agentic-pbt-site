from xarray.core.formatting import maybe_truncate, pretty_print

# Test maybe_truncate with small maxlen values
print("Testing maybe_truncate():")
print("-" * 40)

# Test case 1: maxlen=1 with string length > 1
result = maybe_truncate('00', maxlen=1)
print(f"maybe_truncate('00', maxlen=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 1")
print()

# Test case 2: maxlen=2 with longer string
result = maybe_truncate('hello world', maxlen=2)
print(f"maybe_truncate('hello world', maxlen=2)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 2")
print()

# Test case 3: maxlen=1 with single character (should work)
result = maybe_truncate('a', maxlen=1)
print(f"maybe_truncate('a', maxlen=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 1")
print()

# Test case 4: maxlen=3 with longer string (should work correctly)
result = maybe_truncate('hello world', maxlen=3)
print(f"maybe_truncate('hello world', maxlen=3)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 3")
print()

# Test pretty_print with small numchars values
print("\nTesting pretty_print():")
print("-" * 40)

# Test case 1: numchars=1 with string length > 1
result = pretty_print('00', numchars=1)
print(f"pretty_print('00', numchars=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 1")
print()

# Test case 2: numchars=2 with longer string
result = pretty_print('hello', numchars=2)
print(f"pretty_print('hello', numchars=2)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 2")
print()

# Test case 3: numchars=1 with single character (should work)
result = pretty_print('a', numchars=1)
print(f"pretty_print('a', numchars=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 1")
print()

# Test case 4: numchars=10 with shorter string (should pad)
result = pretty_print('hi', numchars=10)
print(f"pretty_print('hi', numchars=10)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 10")