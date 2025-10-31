from xarray.core.formatting import maybe_truncate, pretty_print

print("Testing maybe_truncate:")
print("=" * 50)

result = maybe_truncate('00', maxlen=1)
print(f"maybe_truncate('00', maxlen=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 1")
print()

result = maybe_truncate('hello world', maxlen=2)
print(f"maybe_truncate('hello world', maxlen=2)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 2")
print()

result = maybe_truncate('a', maxlen=1)
print(f"maybe_truncate('a', maxlen=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 1")
print()

print("Testing pretty_print:")
print("=" * 50)

result = pretty_print('00', numchars=1)
print(f"pretty_print('00', numchars=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 1")
print()

result = pretty_print('hello', numchars=2)
print(f"pretty_print('hello', numchars=2)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 2")
print()

result = pretty_print('a', numchars=1)
print(f"pretty_print('a', numchars=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 1")
print()