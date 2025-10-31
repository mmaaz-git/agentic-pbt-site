#!/usr/bin/env python3
"""Minimal reproduction of pandas.tseries.frequencies reflexivity bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Test various frequency codes
test_freqs = [
    'D', 'h', 'min', 's', 'B', 'C', 'W',
    'M', 'BM', 'Q', 'Q-JAN', 'Y', 'Y-JAN',
]

print("Testing reflexivity: is_superperiod(freq, freq) vs is_subperiod(freq, freq)")
print("=" * 70)
print(f"{'Frequency':<15} {'is_superperiod':<15} {'is_subperiod':<15} {'Consistent?':<15}")
print("-" * 70)

inconsistent_freqs = []

for freq in test_freqs:
    is_super = is_superperiod(freq, freq)
    is_sub = is_subperiod(freq, freq)
    consistent = is_super == is_sub

    if not consistent:
        inconsistent_freqs.append(freq)

    print(f"{freq:<15} {str(is_super):<15} {str(is_sub):<15} {'Yes' if consistent else 'NO':<15}")

if inconsistent_freqs:
    print("\n" + "=" * 70)
    print(f"INCONSISTENT FREQUENCIES FOUND: {', '.join(inconsistent_freqs)}")
    print("=" * 70)

    # Demonstrate the specific issue with annual frequencies
    print("\nDetailed analysis for annual frequency 'Y':")
    print(f"  is_superperiod('Y', 'Y') = {is_superperiod('Y', 'Y')}")
    print(f"  is_subperiod('Y', 'Y') = {is_subperiod('Y', 'Y')}")
    print("\nThis violates the reflexivity property - a frequency should have")
    print("consistent behavior when compared to itself.")