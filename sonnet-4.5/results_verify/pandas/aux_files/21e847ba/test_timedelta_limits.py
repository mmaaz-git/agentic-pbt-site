#!/usr/bin/env python3
"""Test Python timedelta limits to understand when overflow occurs"""

from datetime import timedelta, datetime
import sys

print("Testing Python timedelta limits:")
print(f"timedelta.max = {timedelta.max}")
print(f"timedelta.min = {timedelta.min}")
print(f"timedelta.max.days = {timedelta.max.days}")
print(f"timedelta.max.seconds = {timedelta.max.seconds}")
print(f"timedelta.max.microseconds = {timedelta.max.microseconds}")
print(f"timedelta.max.total_seconds() = {timedelta.max.total_seconds()}")
print()

# Test the exact limit that causes overflow
test_seconds = [
    999999999 * 86400,  # Max days in seconds
    999999999 * 86400 + 86399,  # Max days + max seconds
    1e11,
    1e12,  # This will likely overflow
]

for sec in test_seconds:
    try:
        td = timedelta(seconds=sec)
        print(f"✓ timedelta(seconds={sec:.0e}) succeeded: {td.days} days")
    except OverflowError as e:
        print(f"✗ timedelta(seconds={sec:.0e}) caused OverflowError: {e}")

print("\n" + "="*60 + "\n")

# Test with datetime addition
base_date = datetime(1960, 1, 1)
test_values = [1e10, 1e11, 5e11, 1e12, 1e13, 1e14, 1e15]

print(f"Base date: {base_date}")
print("Testing addition with various second values:")
for val in test_values:
    try:
        result = base_date + timedelta(seconds=val)
        print(f"✓ {val:.0e} seconds -> {result}")
    except OverflowError:
        print(f"✗ {val:.0e} seconds -> OverflowError")

print("\n" + "="*60 + "\n")

# Calculate the exact threshold
max_seconds = timedelta.max.total_seconds()
print(f"Maximum seconds in a timedelta: {max_seconds:.2e}")
print(f"Maximum seconds in scientific notation: {max_seconds}")

# Test values around this threshold
test_around_max = [
    max_seconds - 1,
    max_seconds,
    max_seconds + 1,
]

for val in test_around_max:
    try:
        td = timedelta(seconds=val)
        print(f"✓ timedelta(seconds={val:.15e}) succeeded")
    except OverflowError:
        print(f"✗ timedelta(seconds={val:.15e}) caused OverflowError")