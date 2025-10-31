from pandas.io.excel._util import fill_mi_header

row = [1, None]
control_row = [False, False]

print(f"Input row: {row}")
print(f"Input control_row: {control_row}")

result_row, result_control = fill_mi_header(row.copy(), control_row.copy())

print(f"Output row: {result_row}")
print(f"Output control_row: {result_control}")
print(f"Expected: [1, 1] (forward fill None with 1)")
print(f"Actual: {result_row}")

try:
    assert result_row[1] == 1, f"Expected forward fill, but got {result_row[1]}"
    print("TEST PASSED!")
except AssertionError as e:
    print(f"TEST FAILED: {e}")

# Let's also trace through the logic step by step
print("\n--- Tracing through the logic ---")
row_trace = [1, None]
control_row_trace = [False, False]
last = row_trace[0]
print(f"Initial: last = {last}")

for i in range(1, len(row_trace)):
    print(f"\nIteration {i}:")
    print(f"  row[{i}] = {row_trace[i]}")
    print(f"  control_row[{i}] = {control_row_trace[i]}")

    if not control_row_trace[i]:
        last = row_trace[i]
        print(f"  control_row[{i}] is False, so setting last = row[{i}] = {last}")

    if row_trace[i] == "" or row_trace[i] is None:
        row_trace[i] = last
        print(f"  row[{i}] is None or empty, so setting row[{i}] = last = {last}")
    else:
        control_row_trace[i] = False
        last = row_trace[i]
        print(f"  row[{i}] has value, setting control_row[{i}] = False and last = {row_trace[i]}")

print(f"\nFinal row: {row_trace}")
print(f"Final control_row: {control_row_trace}")