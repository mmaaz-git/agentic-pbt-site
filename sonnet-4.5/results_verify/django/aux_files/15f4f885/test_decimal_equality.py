#!/usr/bin/env python3
import decimal

d1 = decimal.Decimal('1E+1')
d2 = decimal.Decimal('11.0')
d3 = decimal.Decimal('11')

print(f"d1 = {d1} (representation: {repr(d1)})")
print(f"d2 = {d2} (representation: {repr(d2)})")
print(f"d3 = {d3} (representation: {repr(d3)})")
print()
print(f"d1 == d2: {d1 == d2}")
print(f"d2 == d3: {d2 == d3}")
print(f"d1 == d3: {d1 == d3}")
print()
print(f"float(d1) == float(d2): {float(d1) == float(d2)}")
print(f"int(d1) == int(d2): {int(d1) == int(d2)}")