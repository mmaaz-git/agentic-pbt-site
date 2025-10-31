import numpy as np

# Testing the antisymmetry property violation
saturday = np.datetime64('2000-01-01')  # Saturday (non-business day)
monday = np.datetime64('2000-01-03')    # Monday (business day)

# Count business days forward (Saturday to Monday)
count_forward = np.busday_count(saturday, monday)

# Count business days backward (Monday to Saturday)
count_backward = np.busday_count(monday, saturday)

print("Testing busday_count antisymmetry property:")
print("=" * 50)
print(f"Date 1 (Saturday): {saturday}")
print(f"Date 2 (Monday): {monday}")
print()
print(f"busday_count(Saturday, Monday) = {count_forward}")
print(f"busday_count(Monday, Saturday) = {count_backward}")
print()
print("Antisymmetry property check:")
print(f"Expected: count_forward = -count_backward")
print(f"Expected: {count_forward} = {-count_backward}")
print(f"Actual result: {count_forward} {'==' if count_forward == -count_backward else '!='} {-count_backward}")
print()
print(f"Property violated: {count_forward != -count_backward}")

# Let's also test with Tuesday to understand the pattern
tuesday = np.datetime64('2000-01-04')
count_sat_tue = np.busday_count(saturday, tuesday)
count_tue_sat = np.busday_count(tuesday, saturday)

print()
print("Additional test with Tuesday:")
print(f"busday_count(Saturday, Tuesday) = {count_sat_tue}")
print(f"busday_count(Tuesday, Saturday) = {count_tue_sat}")
print(f"Expected: {count_sat_tue} = {-count_tue_sat}")
print(f"Property violated: {count_sat_tue != -count_tue_sat}")