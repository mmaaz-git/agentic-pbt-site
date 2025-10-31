#!/usr/bin/env python3
"""Check what units are supported by parse_timedelta"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from dask.utils import parse_timedelta, format_time

# Check what 'hr' unit is
print("Testing 'hr' vs 'h':")
print(f"  parse_timedelta('1h'): ", end="")
try:
    print(parse_timedelta('1h'))
except Exception as e:
    print(f"ERROR: {e}")

print(f"  parse_timedelta('1hr'): ", end="")
try:
    print(parse_timedelta('1hr'))
except Exception as e:
    print(f"ERROR: {e}")

# Let's look closer at the parsing logic
print("\nAnalyzing parse_timedelta behavior:")
test_cases = [
    "10m1s",    # format_time output style (no space)
    "10m 1s",   # format_time output style (with space)
    "24h0m",    # if format_time used 'h' instead of 'hr'
    "24h 0m",   # if format_time used 'h' with space
]

for s in test_cases:
    print(f"\nInput: '{s}'")
    # Simulate what parse_timedelta does
    s_no_space = s.replace(" ", "")
    print(f"  After removing spaces: '{s_no_space}'")

    # Find last non-alpha character
    for i in range(len(s_no_space) - 1, -1, -1):
        if not s_no_space[i].isalpha():
            break
    index = i + 1

    prefix = s_no_space[:index]
    suffix = s_no_space[index:] or "seconds"

    print(f"  Splits into: prefix='{prefix}', suffix='{suffix}'")
    print(f"  Attempting float('{prefix}'): ", end="")
    try:
        n = float(prefix)
        print(f"OK = {n}")
    except Exception as e:
        print(f"ERROR: {e}")

# Let's check the actual format_time thresholds and outputs
print("\n" + "=" * 60)
print("Understanding format_time thresholds:")
print("=" * 60)

thresholds = [
    (1e-6, "microseconds"),
    (0.001, "milliseconds boundary"),
    (1, "seconds boundary"),
    (10*60, "10 minutes boundary"),
    (10*60 + 1, "just over 10 minutes"),
    (2*60*60, "2 hours boundary"),
    (2*60*60 + 1, "just over 2 hours"),
    (2*24*60*60, "2 days boundary"),
    (2*24*60*60 + 1, "just over 2 days"),
]

for value, desc in thresholds:
    formatted = format_time(value)
    print(f"{desc:30} ({value:10.2f}s) -> '{formatted}'")