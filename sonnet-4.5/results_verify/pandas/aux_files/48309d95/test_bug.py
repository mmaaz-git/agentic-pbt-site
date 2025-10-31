#!/usr/bin/env python3
"""Test the reported infinity comparison bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas.util.version as version_module

def test_infinity_self_comparison():
    inf = version_module.Infinity

    print("Testing Infinity comparisons:")
    print(f"  inf == inf: {inf == inf} (expected: True)")
    print(f"  inf < inf: {inf < inf} (expected: False)")
    print(f"  inf > inf: {inf > inf} (expected: False)")
    print(f"  inf <= inf: {inf <= inf} (expected: True)")
    print(f"  inf >= inf: {inf >= inf} (expected: True)")

    failed = []
    if not (inf == inf):
        failed.append("inf == inf")
    if inf < inf:
        failed.append("inf < inf")
    if inf > inf:
        failed.append("inf > inf")
    if not (inf <= inf):
        failed.append("inf <= inf")
    if not (inf >= inf):
        failed.append("inf >= inf")

    return failed

def test_negative_infinity_self_comparison():
    ninf = version_module.NegativeInfinity

    print("\nTesting NegativeInfinity comparisons:")
    print(f"  ninf == ninf: {ninf == ninf} (expected: True)")
    print(f"  ninf < ninf: {ninf < ninf} (expected: False)")
    print(f"  ninf > ninf: {ninf > ninf} (expected: False)")
    print(f"  ninf <= ninf: {ninf <= ninf} (expected: True)")
    print(f"  ninf >= ninf: {ninf >= ninf} (expected: True)")

    failed = []
    if not (ninf == ninf):
        failed.append("ninf == ninf")
    if ninf < ninf:
        failed.append("ninf < ninf")
    if ninf > ninf:
        failed.append("ninf > ninf")
    if not (ninf <= ninf):
        failed.append("ninf <= ninf")
    if not (ninf >= ninf):
        failed.append("ninf >= ninf")

    return failed

def reproduce_exact_bug():
    """Reproduce exactly what the bug report claims."""
    inf = version_module.Infinity
    ninf = version_module.NegativeInfinity

    print("\nExact bug reproduction (from bug report):")
    print(f"  Infinity > Infinity: {inf > inf} (bug report says: True)")
    print(f"  Infinity <= Infinity: {inf <= inf} (bug report says: False)")
    print(f"  NegativeInfinity < NegativeInfinity: {ninf < ninf} (bug report says: True)")
    print(f"  NegativeInfinity >= NegativeInfinity: {ninf >= ninf} (bug report says: False)")

if __name__ == "__main__":
    inf_failures = test_infinity_self_comparison()
    ninf_failures = test_negative_infinity_self_comparison()
    reproduce_exact_bug()

    print("\n" + "="*50)
    print("SUMMARY:")
    if inf_failures:
        print(f"Infinity failures: {', '.join(inf_failures)}")
    else:
        print("Infinity: All tests passed")

    if ninf_failures:
        print(f"NegativeInfinity failures: {', '.join(ninf_failures)}")
    else:
        print("NegativeInfinity: All tests passed")