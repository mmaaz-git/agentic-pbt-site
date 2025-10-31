#!/usr/bin/env python3
import decimal

d1 = decimal.Decimal('1E+1')
d2 = decimal.Decimal('11.0')
d3 = decimal.Decimal('10')

print(f"d1 value: {d1}")
print(f"d2 value: {d2}")
print(f"d3 value: {d3}")
print()

# Let's check numeric values
print(f"d1 as float: {float(d1)}")
print(f"d2 as float: {float(d2)}")
print(f"d3 as float: {float(d3)}")
print()

# Check if they're numerically equal
print(f"d1 == d2: {d1 == d2}")
print(f"d1 == d3: {d1 == d3}")

# Check the actual numeric comparison
print(f"d1.compare(d2): {d1.compare(d2)}")
print(f"d1.compare(d3): {d1.compare(d3)}")

# Let's see what normalized looks like
print(f"\nd1.normalize(): {d1.normalize()}")
print(f"d2.normalize(): {d2.normalize()}")
print(f"d3.normalize(): {d3.normalize()}")