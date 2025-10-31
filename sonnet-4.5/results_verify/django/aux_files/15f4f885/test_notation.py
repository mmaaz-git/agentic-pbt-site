#!/usr/bin/env python3
import decimal

# 1E+1 means 1 * 10^1 = 10
# So the bug report is wrong about what the values mean

print("1E+1 means 1 * 10^1 =", 1 * 10**1)
print()

d1 = decimal.Decimal('1E+1')
d2 = decimal.Decimal('11.0')

print(f"Decimal('1E+1') = {d1} = {int(d1)}")
print(f"Decimal('11.0') = {d2} = {int(d2)}")
print()

# So the bug is real - 11.0 becomes 10 when processed as float
field_context = decimal.Context(prec=1)
float_val = 11.0
result = field_context.create_decimal_from_float(float_val)
print(f"With prec=1, float {float_val} becomes {result} = {int(result)}")