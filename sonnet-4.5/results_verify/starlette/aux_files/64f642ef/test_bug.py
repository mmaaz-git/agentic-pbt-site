import numpy as np
from hypothesis import given, settings, strategies as st

# First, let's test the specific failing example
print("=== Testing Specific Example ===")
saturday = np.datetime64('2000-01-01')  # January 1, 2000 was a Saturday
monday = np.datetime64('2000-01-03')    # January 3, 2000 was a Monday

count_forward = np.busday_count(saturday, monday)
count_backward = np.busday_count(monday, saturday)

print(f"Saturday: {saturday}")
print(f"Monday: {monday}")
print(f"busday_count(Saturday, Monday) = {count_forward}")
print(f"busday_count(Monday, Saturday) = {count_backward}")
print(f"\nExpected (antisymmetry): {count_forward} = -{count_backward}")
print(f"Actual: {count_forward} != {-count_backward}")
print(f"Antisymmetry violated: {count_forward + count_backward != 0}")

# Let's test more examples to understand the pattern
print("\n=== Testing More Examples ===")
dates = [
    ('2000-01-01', 'Saturday'),  # Saturday
    ('2000-01-02', 'Sunday'),    # Sunday
    ('2000-01-03', 'Monday'),    # Monday
    ('2000-01-04', 'Tuesday'),   # Tuesday
    ('2000-01-05', 'Wednesday')  # Wednesday
]

for i in range(len(dates)):
    for j in range(len(dates)):
        if i != j:
            d1 = np.datetime64(dates[i][0])
            d2 = np.datetime64(dates[j][0])
            forward = np.busday_count(d1, d2)
            backward = np.busday_count(d2, d1)
            print(f"{dates[i][1][:3]} -> {dates[j][1][:3]}: forward={forward:2d}, backward={backward:2d}, sum={forward+backward:2d}")

# Now let's run the hypothesis test
print("\n=== Running Hypothesis Test ===")
datetime_strategy = st.integers(min_value=0, max_value=20000).map(
    lambda days: np.datetime64('2000-01-01') + np.timedelta64(days, 'D')
)

failed_cases = []

@given(datetime_strategy, datetime_strategy)
@settings(max_examples=1000)
def test_busday_count_antisymmetric(date1, date2):
    count_forward = np.busday_count(date1, date2)
    count_backward = np.busday_count(date2, date1)
    if count_forward != -count_backward:
        failed_cases.append((date1, date2, count_forward, count_backward))

# Run the test
try:
    test_busday_count_antisymmetric()
    print("Hypothesis test completed")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

if failed_cases:
    print(f"\nFound {len(failed_cases)} failing cases. First 10:")
    for i, (d1, d2, fwd, bwd) in enumerate(failed_cases[:10]):
        print(f"  {d1} -> {d2}: forward={fwd}, backward={bwd}, sum={fwd+bwd}")
else:
    print("No failing cases found (unexpected!)")