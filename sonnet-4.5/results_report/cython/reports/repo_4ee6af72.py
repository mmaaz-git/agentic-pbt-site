from Cython.Compiler.PyrexTypes import cap_length

result = cap_length('00000000000', max_len=10)
print(f"Result: {result!r}")
print(f"Length: {len(result)}")
print(f"Expected max: 10")
print(f"Violates constraint: {len(result) > 10}")