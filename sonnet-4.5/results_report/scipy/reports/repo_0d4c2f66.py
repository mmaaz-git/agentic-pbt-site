#!/usr/bin/env python3
"""Minimal reproduction case for scipy.io.arff DateAttribute bug."""

from scipy.io.arff._arffread import DateAttribute

# Test case 1: Pattern with only month (should have unit='M')
print("Test 1: date 'MM'")
date_format, datetime_unit = DateAttribute._get_date_format("date 'MM'")
print(f"Format: {date_format}, Unit: {datetime_unit}")
print(f"Expected: Format: %m, Unit: M")
print(f"Bug causes unit to be incorrectly 'Y' instead of 'M'? {datetime_unit == 'Y'}")
print()

# Test case 2: Empty pattern (should raise ValueError)
print("Test 2: date ''")
try:
    date_format, datetime_unit = DateAttribute._get_date_format("date ''")
    print(f"Format: {date_format}, Unit: {datetime_unit}")
    print("ERROR: Should have raised ValueError for invalid format")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()

# Test case 3: Invalid pattern with no date components
print("Test 3: date 'XXX'")
try:
    date_format, datetime_unit = DateAttribute._get_date_format("date 'XXX'")
    print(f"Format: {date_format}, Unit: {datetime_unit}")
    print("ERROR: Should have raised ValueError for invalid format")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()

# Test case 4: Pattern with day only
print("Test 4: date 'dd'")
date_format, datetime_unit = DateAttribute._get_date_format("date 'dd'")
print(f"Format: {date_format}, Unit: {datetime_unit}")
print(f"Expected: Format: %d, Unit: D")
print()

# Test case 5: Pattern with time components only
print("Test 5: date 'HH:mm:ss'")
date_format, datetime_unit = DateAttribute._get_date_format("date 'HH:mm:ss'")
print(f"Format: {date_format}, Unit: {datetime_unit}")
print(f"Expected: Format: %H:%M:%S, Unit: s")
print()

# Test case 6: Complex pattern without year
print("Test 6: date 'MM-dd HH:mm'")
date_format, datetime_unit = DateAttribute._get_date_format("date 'MM-dd HH:mm'")
print(f"Format: {date_format}, Unit: {datetime_unit}")
print(f"Expected: Format: %m-%d %H:%M, Unit: m")