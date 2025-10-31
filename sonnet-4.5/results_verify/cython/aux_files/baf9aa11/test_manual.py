from Cython.Utils import normalise_float_repr

test_cases = [
    "-1e-10",
    "-0.00001",
    "-1.5e-5",
    "-1.1754943508222875e-38",
]

for float_str in test_cases:
    result = normalise_float_repr(float_str)
    print(f"{float_str:30} -> {result}")
    try:
        float(result)
    except ValueError:
        print(f"  ERROR: '{result}' is not a valid float!")