#!/usr/bin/env python3
"""Test script to reproduce the AlwaysGreaterThan/AlwaysLessThan bug"""

from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

print("Testing AlwaysGreaterThan:")
print("-" * 40)
inf1 = AlwaysGreaterThan()
inf2 = AlwaysGreaterThan()

print(f"inf1 == inf2: {inf1 == inf2}")  # Expected: True
print(f"inf1 != inf2: {inf1 != inf2}")  # Expected: False
print(f"inf1 > inf2: {inf1 > inf2}")    # Expected: False (since they're equal)
print(f"inf1 < inf2: {inf1 < inf2}")    # Expected: False (since they're equal)
print(f"inf1 >= inf2: {inf1 >= inf2}")  # Expected: True (since they're equal)
print(f"inf1 <= inf2: {inf1 <= inf2}")  # Expected: True (since they're equal)

print("\nTesting AlwaysLessThan:")
print("-" * 40)
ninf1 = AlwaysLessThan()
ninf2 = AlwaysLessThan()

print(f"ninf1 == ninf2: {ninf1 == ninf2}")  # Expected: True
print(f"ninf1 != ninf2: {ninf1 != ninf2}")  # Expected: False
print(f"ninf1 > ninf2: {ninf1 > ninf2}")    # Expected: False (since they're equal)
print(f"ninf1 < ninf2: {ninf1 < ninf2}")    # Expected: False (since they're equal)
print(f"ninf1 >= ninf2: {ninf1 >= ninf2}")  # Expected: True (since they're equal)
print(f"ninf1 <= ninf2: {ninf1 <= ninf2}")  # Expected: True (since they're equal)

print("\nRunning property-based tests:")
print("-" * 40)

def test_always_greater_than_equals_itself():
    inf1 = AlwaysGreaterThan()
    inf2 = AlwaysGreaterThan()
    try:
        assert inf1 == inf2
        print("✓ inf1 == inf2")
    except AssertionError:
        print("✗ inf1 == inf2 FAILED")

    try:
        assert not (inf1 != inf2)
        print("✓ not (inf1 != inf2)")
    except AssertionError:
        print("✗ not (inf1 != inf2) FAILED")

    try:
        assert not (inf1 < inf2)
        print("✓ not (inf1 < inf2)")
    except AssertionError:
        print("✗ not (inf1 < inf2) FAILED")

    try:
        assert not (inf1 > inf2)
        print("✓ not (inf1 > inf2)")
    except AssertionError:
        print("✗ not (inf1 > inf2) FAILED - BUG CONFIRMED!")

def test_always_less_than_equals_itself():
    ninf1 = AlwaysLessThan()
    ninf2 = AlwaysLessThan()
    try:
        assert ninf1 == ninf2
        print("✓ ninf1 == ninf2")
    except AssertionError:
        print("✗ ninf1 == ninf2 FAILED")

    try:
        assert not (ninf1 != ninf2)
        print("✓ not (ninf1 != ninf2)")
    except AssertionError:
        print("✗ not (ninf1 != ninf2) FAILED")

    try:
        assert not (ninf1 < ninf2)
        print("✓ not (ninf1 < ninf2)")
    except AssertionError:
        print("✗ not (ninf1 < ninf2) FAILED - BUG CONFIRMED!")

    try:
        assert not (ninf1 > ninf2)
        print("✓ not (ninf1 > ninf2)")
    except AssertionError:
        print("✗ not (ninf1 > ninf2) FAILED")

print("\nAlwaysGreaterThan tests:")
test_always_greater_than_equals_itself()

print("\nAlwaysLessThan tests:")
test_always_less_than_equals_itself()