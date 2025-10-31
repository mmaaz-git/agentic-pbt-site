from pandas.io.excel._util import fill_mi_header

# Test case demonstrating the bug
row = [1, None]
control_row = [False, False]

print(f"Input row: {row}")
print(f"Input control_row: {control_row}")

result_row, result_control = fill_mi_header(row.copy(), control_row.copy())

print(f"\nOutput row: {result_row}")
print(f"Output control_row: {result_control}")

print(f"\nExpected row: [1, 1] (forward fill None with 1)")
print(f"Actual row: {result_row}")

# Check if the forward fill worked correctly
if result_row[1] == 1:
    print("\n✓ Forward fill worked correctly")
else:
    print(f"\n✗ Forward fill failed: Expected row[1]=1, but got row[1]={result_row[1]}")

# Additional test case with multiple Nones
print("\n" + "="*50)
print("Additional test case with multiple None values:")

row2 = [5, None, None, 10, None]
control_row2 = [False, False, False, False, False]

print(f"\nInput row: {row2}")
print(f"Input control_row: {control_row2}")

result_row2, result_control2 = fill_mi_header(row2.copy(), control_row2.copy())

print(f"\nOutput row: {result_row2}")
print(f"Output control_row: {result_control2}")

print(f"\nExpected row: [5, 5, 5, 10, 10] (forward fill Nones)")
print(f"Actual row: {result_row2}")

# Check if all Nones were filled
expected2 = [5, 5, 5, 10, 10]
if result_row2 == expected2:
    print("\n✓ All forward fills worked correctly")
else:
    print(f"\n✗ Forward fill failed: Expected {expected2}, but got {result_row2}")