#!/usr/bin/env python3
from scipy.constants import convert_temperature
import math

# Test all scale variants to ensure they all have the same issue
scales_variants = {
    'Celsius': ['Celsius', 'celsius', 'C', 'c'],
    'Kelvin': ['Kelvin', 'kelvin', 'K', 'k'],
    'Fahrenheit': ['Fahrenheit', 'fahrenheit', 'F', 'f'],
    'Rankine': ['Rankine', 'rankine', 'R', 'r']
}

test_val = 1e-20
print("Testing all scale name variants for identity conversion:")
print("=" * 60)

for scale_name, variants in scales_variants.items():
    print(f"\n{scale_name} variants:")
    for var1 in variants:
        for var2 in variants:
            result = convert_temperature(test_val, var1, var2)
            if result != test_val:
                print(f"  {var1} -> {var2}: LOSES PRECISION (result={result})")
            else:
                print(f"  {var1} -> {var2}: OK")

# Now let's understand WHY some scales preserve and others don't
print("\n" + "=" * 60)
print("Understanding the conversion paths:\n")

# Look at the conversion formulas
print("Conversion logic from source code:")
print("-" * 40)
print("Step 1: Convert to Kelvin")
print("  Celsius: tempo = val + 273.15")
print("  Kelvin: tempo = val")
print("  Fahrenheit: tempo = (val - 32) * 5/9 + 273.15")
print("  Rankine: tempo = val * 5/9")
print("\nStep 2: Convert from Kelvin")
print("  to Celsius: res = tempo - 273.15")
print("  to Kelvin: res = tempo")
print("  to Fahrenheit: res = (tempo - 273.15) * 9/5 + 32")
print("  to Rankine: res = tempo * 9/5")

print("\n" + "-" * 40)
print("Analysis of identity conversions:\n")

# Celsius -> Celsius
val = 1e-20
print(f"Celsius -> Celsius with val={val}:")
print(f"  Step 1: tempo = {val} + 273.15 = {val + 273.15}")
print(f"  Step 2: res = {val + 273.15} - 273.15 = {(val + 273.15) - 273.15}")
print(f"  Problem: Adding/subtracting 273.15 loses precision for small values\n")

# Kelvin -> Kelvin
print(f"Kelvin -> Kelvin with val={val}:")
print(f"  Step 1: tempo = {val}")
print(f"  Step 2: res = tempo = {val}")
print(f"  No arithmetic operations, so no precision loss\n")

# Fahrenheit -> Fahrenheit
print(f"Fahrenheit -> Fahrenheit with val={val}:")
tempo_f = (val - 32) * 5/9 + 273.15
res_f = (tempo_f - 273.15) * 9/5 + 32
print(f"  Step 1: tempo = ({val} - 32) * 5/9 + 273.15 = {tempo_f}")
print(f"  Step 2: res = ({tempo_f} - 273.15) * 9/5 + 32 = {res_f}")
print(f"  Multiple operations with large constants cause precision loss\n")

# Rankine -> Rankine
print(f"Rankine -> Rankine with val={val}:")
tempo_r = val * 5/9
res_r = tempo_r * 9/5
print(f"  Step 1: tempo = {val} * 5/9 = {tempo_r}")
print(f"  Step 2: res = {tempo_r} * 9/5 = {res_r}")
print(f"  Only multiplication by reciprocal fractions, minimal precision loss")