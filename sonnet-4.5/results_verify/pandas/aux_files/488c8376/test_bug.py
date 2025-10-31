#!/usr/bin/env python3
"""Test script to reproduce the reported bug."""

from pandas.tseries.frequencies import is_subperiod, is_superperiod
from hypothesis import given, strategies as st, settings
import traceback

print("=" * 60)
print("Testing is_subperiod/is_superperiod symmetry")
print("=" * 60)

# First, test the specific case mentioned in the bug report
print("\n1. Testing specific case: source='Y', target='Y'")
source = 'Y'
target = 'Y'

sub_result = is_subperiod(source, target)
super_result = is_superperiod(target, source)

print(f"is_subperiod('Y', 'Y') = {sub_result}")
print(f"is_superperiod('Y', 'Y') = {super_result}")

if sub_result == super_result:
    print("✓ Symmetry holds for Y->Y")
else:
    print("✗ Symmetry violated for Y->Y!")

# Test other same-frequency cases
print("\n2. Testing other same-frequency cases:")
test_cases = ['Q', 'M', 'W', 'D', 'h', 'min', 's', 'ms', 'us', 'ns']
violations = []

for freq in test_cases:
    sub = is_subperiod(freq, freq)
    sup = is_superperiod(freq, freq)
    if sub != sup:
        violations.append((freq, sub, sup))
        print(f"  {freq}: is_subperiod={sub}, is_superperiod={sup} - VIOLATION")
    else:
        print(f"  {freq}: is_subperiod={sub}, is_superperiod={sup} - OK")

# Now test with hypothesis
print("\n3. Running property-based test with Hypothesis:")

VALID_FREQ_STRINGS = [
    "Y", "Q", "M", "W", "D", "B", "C", "h", "min", "s", "ms", "us", "ns",
    "Y-JAN", "Y-FEB", "Y-MAR", "Q-JAN", "Q-FEB", "W-MON", "W-TUE"
]

violation_count = 0
test_count = 0

@given(
    source=st.sampled_from(VALID_FREQ_STRINGS),
    target=st.sampled_from(VALID_FREQ_STRINGS)
)
@settings(max_examples=1000)
def test_subperiod_superperiod_symmetry(source, target):
    global violation_count, test_count
    test_count += 1

    sub_result = is_subperiod(source, target)
    super_result = is_superperiod(target, source)

    if sub_result != super_result:
        violation_count += 1
        if violation_count <= 5:  # Only print first 5 violations
            print(f"  Violation found: is_subperiod({source!r}, {target!r})={sub_result}, "
                  f"is_superperiod({target!r}, {source!r})={super_result}")

try:
    test_subperiod_superperiod_symmetry()
    print(f"\nCompleted {test_count} tests")
    print(f"Found {violation_count} symmetry violations")
except Exception as e:
    print(f"\nTest failed with exception: {e}")
    traceback.print_exc()

# Test the relationship more broadly
print("\n4. Testing the semantic meaning:")
print("For downsampling/upsampling, these should be symmetric:")
test_pairs = [
    ('M', 'Y'),  # Month to Year
    ('D', 'M'),  # Day to Month
    ('h', 'D'),  # Hour to Day
    ('s', 'h'),  # Second to Hour
]

for source, target in test_pairs:
    sub = is_subperiod(source, target)
    sup = is_superperiod(target, source)
    print(f"  is_subperiod({source}, {target})={sub}, is_superperiod({target}, {source})={sup}")
    if sub != sup:
        print(f"    ^ VIOLATION: These should be equal!")