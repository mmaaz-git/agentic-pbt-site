#!/usr/bin/env python3
"""Test reproduction of the attrs to_bool bug report"""

from attrs import converters

print("=" * 60)
print("Testing attrs.converters.to_bool behavior")
print("=" * 60)

print("\nDocumented behavior:")
print(f"to_bool(1) = {converters.to_bool(1)}")
print(f"to_bool(0) = {converters.to_bool(0)}")
print(f"to_bool(True) = {converters.to_bool(True)}")
print(f"to_bool(False) = {converters.to_bool(False)}")
print(f"to_bool('true') = {converters.to_bool('true')}")
print(f"to_bool('false') = {converters.to_bool('false')}")

print("\nUndocumented behavior (float inputs):")
print(f"to_bool(1.0) = {converters.to_bool(1.0)}")
print(f"to_bool(0.0) = {converters.to_bool(0.0)}")

print("\nInconsistent behavior (other float values):")
try:
    result = converters.to_bool(1.5)
    print(f"to_bool(1.5) = {result}")
except ValueError as e:
    print(f"to_bool(1.5) raises ValueError: {e}")

try:
    result = converters.to_bool(2.0)
    print(f"to_bool(2.0) = {result}")
except ValueError as e:
    print(f"to_bool(2.0) raises ValueError: {e}")

try:
    result = converters.to_bool(0.5)
    print(f"to_bool(0.5) = {result}")
except ValueError as e:
    print(f"to_bool(0.5) raises ValueError: {e}")

print("\nTesting numeric equality in Python:")
print(f"1.0 == 1: {1.0 == 1}")
print(f"0.0 == 0: {0.0 == 0}")
print(f"1.5 == 1: {1.5 == 1}")
print(f"2.0 == 2: {2.0 == 2}")