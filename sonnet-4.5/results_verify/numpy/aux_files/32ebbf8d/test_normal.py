#!/usr/bin/env python3
import numpy.f2py.symbolic as symbolic

print("Testing properly quoted strings...")
test_cases = [
    "'hello'",
    '"world"',
    "'test string'",
    '"another test"',
    "no quotes",
    "'nested \"quotes\"'",
    '"nested \'quotes\'"',
    "text 'with quotes' inside",
    'text "with other" quotes',
]

for test in test_cases:
    print(f"\nInput: {repr(test)}")
    try:
        s_no_quotes, d = symbolic.eliminate_quotes(test)
        print(f"  Output: {repr(s_no_quotes)}")
        print(f"  Dictionary: {d}")

        # Test round-trip
        s_restored = symbolic.insert_quotes(s_no_quotes, d)
        print(f"  Restored: {repr(s_restored)}")
        print(f"  Round-trip success: {test == s_restored}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")