from pandas.io.formats.format import _trim_zeros_float

# Let's trace through the algorithm step by step for a specific example
inputs = ['0.00', '0.010', '0.0000']

print("Original input:", inputs)
print("\nStep-by-step trace of the algorithm:")

# Simulate the algorithm manually
trimmed = inputs.copy()
step = 0

import re
decimal = "."
number_regex = re.compile(rf"^\s*[\+-]?[0-9]+\{decimal}[0-9]*$")

def is_number_with_decimal(x):
    return re.match(number_regex, x) is not None

def should_trim(values):
    numbers = [x for x in values if is_number_with_decimal(x)]
    return len(numbers) > 0 and all(x.endswith("0") for x in numbers)

while should_trim(trimmed):
    step += 1
    print(f"Step {step}:")
    print(f"  Current: {trimmed}")
    print(f"  Numbers with decimals: {[x for x in trimmed if is_number_with_decimal(x)]}")
    print(f"  All end with '0': {all(x.endswith('0') for x in [x for x in trimmed if is_number_with_decimal(x)])}")
    trimmed = [x[:-1] if is_number_with_decimal(x) else x for x in trimmed]
    print(f"  After trim: {trimmed}")

# Add '0' if ends with decimal
result = [
    x + "0" if is_number_with_decimal(x) and x.endswith(decimal) else x
    for x in trimmed
]

print(f"\nFinal result: {result}")

# Now run the actual function
actual_result = _trim_zeros_float(inputs)
print(f"Actual function result: {actual_result}")

# Test with different numbers having same trailing zeros
print("\n" + "="*50)
print("Testing with same number of trailing zeros:")
same_zeros = ['1.200', '2.300', '3.400']
result_same = _trim_zeros_float(same_zeros)
print(f"Input:  {same_zeros}")
print(f"Output: {result_same}")
print(f"Decimal lengths: {[len(r.split('.')[1]) for r in result_same]}")

print("\n" + "="*50)
print("Testing with different number of trailing zeros:")
diff_zeros = ['1.20', '2.300', '3.4000']
result_diff = _trim_zeros_float(diff_zeros)
print(f"Input:  {diff_zeros}")
print(f"Output: {result_diff}")
print(f"Decimal lengths: {[len(r.split('.')[1]) for r in result_diff]}")

print("\n" + "="*50)
print("Testing the meaning of 'equally':")
print("If 'equally' means 'same amount trimmed from each':")
print("  ['1.200', '2.3000'] -> trim 2 zeros -> ['1.2', '2.30']")
print("If 'equally' means 'trim until all have same length':")
print("  ['1.200', '2.3000'] -> ??? (unclear)")
print("If 'equally' means 'trim as much as possible from all':")
print("  ['1.200', '2.3000'] -> trim until one can't be trimmed -> ['1.2', '2.30']")

actual = _trim_zeros_float(['1.200', '2.3000'])
print(f"Actual result: {actual}")
print(f"Decimal lengths: {[len(r.split('.')[1]) for r in actual]}")