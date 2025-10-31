#!/usr/bin/env python3
"""Test if the original packaging library has the same behavior."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

# Try to import from packaging library
try:
    from packaging._structures import Infinity, NegativeInfinity
    print("Using packaging library version")
except ImportError:
    print("packaging._structures not available")
    sys.exit(1)

def test_packaging_infinity():
    print("Testing packaging library Infinity:")
    print(f"  Infinity > Infinity: {Infinity > Infinity}")
    print(f"  Infinity <= Infinity: {Infinity <= Infinity}")
    print(f"  Infinity == Infinity: {Infinity == Infinity}")
    print(f"  Infinity >= Infinity: {Infinity >= Infinity}")
    print(f"  Infinity < Infinity: {Infinity < Infinity}")

def test_packaging_negative_infinity():
    print("\nTesting packaging library NegativeInfinity:")
    print(f"  NegativeInfinity < NegativeInfinity: {NegativeInfinity < NegativeInfinity}")
    print(f"  NegativeInfinity >= NegativeInfinity: {NegativeInfinity >= NegativeInfinity}")
    print(f"  NegativeInfinity == NegativeInfinity: {NegativeInfinity == NegativeInfinity}")
    print(f"  NegativeInfinity <= NegativeInfinity: {NegativeInfinity <= NegativeInfinity}")
    print(f"  NegativeInfinity > NegativeInfinity: {NegativeInfinity > NegativeInfinity}")

if __name__ == "__main__":
    test_packaging_infinity()
    test_packaging_negative_infinity()