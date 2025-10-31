import sys
import os
# Add Django path to sys.path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Simple implementation to test the function
def split_tzname_delta(tzname):
    """
    Split a time zone name into a 3-tuple of (name, sign, offset).
    This is a copy of the Django function for testing.
    """
    from django.utils.dateparse import parse_time
    for sign in ["+", "-"]:
        if sign in tzname:
            name, offset = tzname.rsplit(sign, 1)
            if offset and parse_time(offset):
                if ":" not in offset:
                    offset = f"{offset}:00"
                return name, sign, offset
    return tzname, None, None


# Test case 1: HHMM format without colon
input_tz = 'UTC+0530'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 1 - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"Expected offset: '05:30', Actual offset: {offset!r}")
print()

# Test case 2: Edge case with 0000
input_tz = 'A+0000'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 2 - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"Expected offset: '00:00', Actual offset: {offset!r}")
print()

# Test case 3: Another HHMM case
input_tz = 'EST-0245'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 3 - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"Expected offset: '02:45', Actual offset: {offset!r}")
print()

# Test case 4: Valid input with HH format (2 digits)
input_tz = 'UTC+05'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 4 - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"Expected offset: '05:00', Actual offset: {offset!r}")
print()

# Test case 5: Correct format with colon works
input_tz = 'UTC+05:30'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 5 (correct format) - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"This format works correctly!")
print()

# Show what PostgreSQL would receive
print("=" * 60)
print("PostgreSQL AT TIME ZONE impact:")
print("=" * 60)

def prepare_tzname_delta(tzname):
    """Mimics PostgreSQL backend's _prepare_tzname_delta."""
    name, sign, offset = split_tzname_delta(tzname)
    if offset:
        sign = "-" if sign == "+" else "+"
        return f"{name}{sign}{offset}"
    return tzname

for test_tz in ['UTC+0530', 'EST-0245', 'A+0000']:
    prepared = prepare_tzname_delta(test_tz)
    print(f"Input: {test_tz:15} -> Prepared for PostgreSQL: {prepared}")
    print(f"  Expected format: {test_tz.split('+')[0] if '+' in test_tz else test_tz.split('-')[0]}{'-' if '+' in test_tz else '+'}{test_tz[-2:] if len(test_tz.split('+' if '+' in test_tz else '-')[-1]) == 4 else test_tz[-2:]}:{test_tz[-2:] if len(test_tz.split('+' if '+' in test_tz else '-')[-1]) == 4 else '00'}")
    if '0530:00' in prepared or '0245:00' in prepared or '0000:00' in prepared:
        print(f"  ❌ INVALID: PostgreSQL expects HH:MM format, not HHMM:00")
    print()