#!/usr/bin/env python3
"""Test to see how apply_filters is typically called"""

# Test how the caller checks filters
filters = []
stats = [{'columns': []}]

print("Checking how apply_filters would be called:")
print(f"  filters = {filters}")
print(f"  stats = {stats}")
print(f"  bool(filters) = {bool(filters)}")
print(f"  bool(stats) = {bool(stats)}")
print(f"  filters and stats = {filters and stats}")

if filters and stats:
    print("  -> Would call apply_filters")
else:
    print("  -> Would NOT call apply_filters (skipped)")

print()

filters = [('x', '>', 5)]
print("With non-empty filters:")
print(f"  filters = {filters}")
print(f"  bool(filters) = {bool(filters)}")
print(f"  filters and stats = {bool(filters and stats)}")

if filters and stats:
    print("  -> Would call apply_filters")
else:
    print("  -> Would NOT call apply_filters")