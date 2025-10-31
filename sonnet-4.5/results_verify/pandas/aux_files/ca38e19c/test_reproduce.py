from pandas.io.formats.format import _trim_zeros_float

# Test case from bug report
inputs = ['0.00', '0.0000', '0.00000']
result = _trim_zeros_float(inputs)

print(f"Input:  {inputs}")
print(f"Output: {result}")

decimal_lengths = [len(r.split('.')[1]) for r in result]
print(f"Decimal lengths: {decimal_lengths}")

# Check if all have same decimal lengths
if len(set(decimal_lengths)) == 1:
    print("All decimal lengths are equal")
else:
    print("Decimal lengths are NOT equal")

# Test some other cases to understand the behavior
test_cases = [
    ['1.00', '2.00', '3.00'],
    ['1.10', '2.20', '3.30'],
    ['1.100', '2.200', '3.300'],
    ['1.1000', '2.2000', '3.3000'],
    ['1.00', '2.000', '3.0000'],
    ['1.0', '2.0', '3.0'],
    ['1.10', '2.200', '3.3000'],
    ['0.50', '0.5000', '0.50000']
]

print("\nAdditional test cases:")
for test in test_cases:
    result = _trim_zeros_float(test)
    decimal_lengths = [len(r.split('.')[1]) for r in result]
    print(f"Input:  {test}")
    print(f"Output: {result}")
    print(f"Decimal lengths: {decimal_lengths}")
    print(f"All equal: {len(set(decimal_lengths)) == 1}")
    print()