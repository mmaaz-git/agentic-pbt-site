from pandas.io.excel._util import _range2cols

print("Testing _range2cols with forward and reversed ranges:")
print("=" * 60)

result_forward = _range2cols("A:C")
print(f"Forward range 'A:C' returns: {result_forward}")
print(f"  Expected: [0, 1, 2] (columns A, B, C)")
print(f"  Length: {len(result_forward)}")
print()

result_reversed = _range2cols("C:A")
print(f"Reversed range 'C:A' returns: {result_reversed}")
print(f"  Expected: [0, 1, 2] or [2, 1, 0] (columns C, B, A or A, B, C)")
print(f"  Length: {len(result_reversed)}")
print()

result_larger_reversed = _range2cols("AA:A")
print(f"Reversed range 'AA:A' returns: {result_larger_reversed}")
print(f"  Expected: list containing 27 elements (columns A through AA)")
print(f"  Length: {len(result_larger_reversed)}")
print()

result_complex = _range2cols("D:B,F,Z:AB")
print(f"Complex range 'D:B,F,Z:AB' returns: {result_complex}")
print(f"  Note: 'D:B' is reversed, should return columns B, C, D")
print(f"  Expected: B, C, D (1,2,3) + F (5) + Z, AA, AB (25,26,27)")
print(f"  Length: {len(result_complex)}")