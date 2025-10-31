#!/usr/bin/env python3
"""Reproduce the bug reported in AlwaysGreaterThan/AlwaysLessThan"""

from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan
import numpy as np

print("=== Testing AlwaysGreaterThan ===")
agt = AlwaysGreaterThan()
print(f"agt > agt: {agt > agt}  (should be False for irreflexivity)")
print(f"agt == agt: {agt == agt}")
print(f"agt < agt: {agt < agt}")
print(f"agt >= agt: {agt >= agt}")
print(f"agt <= agt: {agt <= agt}")

print("\n=== Testing AlwaysLessThan ===")
alt = AlwaysLessThan()
print(f"alt < alt: {alt < alt}  (should be False for irreflexivity)")
print(f"alt == alt: {alt == alt}")
print(f"alt > alt: {alt > alt}")
print(f"alt <= alt: {alt <= alt}")
print(f"alt >= alt: {alt >= alt}")

print("\n=== Testing two AlwaysGreaterThan instances (trichotomy) ===")
agt1 = AlwaysGreaterThan()
agt2 = AlwaysGreaterThan()
print(f"agt1 > agt2: {agt1 > agt2}")
print(f"agt1 == agt2: {agt1 == agt2}")
print(f"agt1 < agt2: {agt1 < agt2}")
less = agt1 < agt2
equal = agt1 == agt2
greater = agt1 > agt2
true_count = sum([less, equal, greater])
print(f"Number of True conditions (should be exactly 1): {true_count}")
if true_count != 1:
    print("TRICHOTOMY VIOLATED!")

print("\n=== Testing two AlwaysLessThan instances (trichotomy) ===")
alt1 = AlwaysLessThan()
alt2 = AlwaysLessThan()
print(f"alt1 < alt2: {alt1 < alt2}")
print(f"alt1 == alt2: {alt1 == alt2}")
print(f"alt1 > alt2: {alt1 > alt2}")
less = alt1 < alt2
equal = alt1 == alt2
greater = alt1 > alt2
true_count = sum([less, equal, greater])
print(f"Number of True conditions (should be exactly 1): {true_count}")
if true_count != 1:
    print("TRICHOTOMY VIOLATED!")

print("\n=== Comparison with np.inf for reference ===")
print(f"np.inf > np.inf: {np.inf > np.inf}")
print(f"np.inf == np.inf: {np.inf == np.inf}")
print(f"np.inf < np.inf: {np.inf < np.inf}")
print(f"-np.inf < -np.inf: {-np.inf < -np.inf}")
print(f"-np.inf == -np.inf: {-np.inf == -np.inf}")
print(f"-np.inf > -np.inf: {-np.inf > -np.inf}")