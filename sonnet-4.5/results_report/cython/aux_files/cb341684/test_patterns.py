from Cython.Utils import normalise_float_repr

# Test various patterns
test_cases = [
    "1.23e-5",
    "1.234e-5",
    "1.2345e-5",
    "1.23456e-5",
    "1.234567e-5",
    "1.2345678e-5",
    "5.960464477539063e-08",  # From hypothesis
    "6.103515625e-05",  # From bug report
]

for x in test_cases:
    result = normalise_float_repr(x)
    input_val = float(x)
    output_val = float(result)
    error = abs(output_val - input_val) / abs(input_val) if input_val != 0 else 0

    print(f"Input: {x:20s} -> Output: {result:20s}")
    print(f"  Values: {input_val:.15e} -> {output_val:.15e}")
    print(f"  Error factor: {error:.2e}")
    print()