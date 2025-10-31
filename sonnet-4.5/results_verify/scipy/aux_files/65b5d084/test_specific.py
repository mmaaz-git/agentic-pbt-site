#!/usr/bin/env python3
"""Test specific edge cases for the bug"""

from scipy.io.arff._arffread import DateAttribute

# Test patterns that expose the bug
test_patterns = [
    "date H",      # Only hour
    "date m",      # Only minute
    "date s",      # Only second
    "date HH",     # Hour format
    "date mm",     # Minute format
    "date ss",     # Second format
]

print("Testing patterns that might expose the bug:")
for pattern in test_patterns:
    try:
        fmt, unit = DateAttribute._get_date_format(pattern)
        print(f"  {pattern:<20} -> unit={unit}, fmt={fmt}")
        if 'y' not in pattern.lower() and unit == "Y":
            print(f"    BUG EXPOSED: Pattern has no year but unit is 'Y'")
    except ValueError as e:
        print(f"  {pattern:<20} -> ValueError: {e}")

# Now let's trace through the code logic for "date HH"
print("\nTracing logic for 'date HH':")
pattern = "date HH"
print(f"1. Pattern: {pattern}")
print(f"2. After matching, pattern becomes 'HH'")
print(f"3. Check 'yyyy' in 'HH': {('yyyy' in 'HH')} -> False, skip")
print(f"4. Check 'yy' (BUG!): {bool('yy')} -> True, always enters!")
print(f"5. Replace 'yy' in 'HH': {'HH'.replace('yy', '%y')} -> No change")
print(f"6. Set datetime_unit = 'Y' (WRONG!)")
print(f"7. Check 'MM' in 'HH': {('MM' in 'HH')} -> False, skip")
print(f"8. Check 'dd' in 'HH': {('dd' in 'HH')} -> False, skip")
print(f"9. Check 'HH' in 'HH': {('HH' in 'HH')} -> True")
print(f"10. Replace 'HH' with '%H' and set datetime_unit = 'h'")
print(f"11. Final: datetime_unit = 'h' (overwrites incorrect 'Y')")

print("\nNow tracing for 'date H' (single H):")
pattern = "date H"
print(f"1. Pattern: {pattern}")
print(f"2. After matching, pattern becomes 'H'")
print(f"3. Check 'yyyy' in 'H': {('yyyy' in 'H')} -> False, skip")
print(f"4. Check 'yy' (BUG!): {bool('yy')} -> True, always enters!")
print(f"5. Replace 'yy' in 'H': {'H'.replace('yy', '%y')} -> No change")
print(f"6. Set datetime_unit = 'Y' (WRONG!)")
print(f"7. Check 'MM' in 'H': {('MM' in 'H')} -> False, skip")
print(f"8. Check 'dd' in 'H': {('dd' in 'H')} -> False, skip")
print(f"9. Check 'HH' in 'H': {('HH' in 'H')} -> False, skip")
print(f"10. Check 'mm' in 'H': {('mm' in 'H')} -> False, skip")
print(f"11. Check 'ss' in 'H': {('ss' in 'H')} -> False, skip")
print(f"12. datetime_unit is still 'Y' (BUG EXPOSED!)")