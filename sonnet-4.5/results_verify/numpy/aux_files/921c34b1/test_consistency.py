import numpy as np
import numpy.ma as ma

print("Testing mask consistency patterns:")
print("="*60)

# Test 1: Check if explicit mask=[] is treated differently than no mask
print("\n1. Empty array with explicit empty mask vs no mask:")
arr1 = ma.masked_array([], mask=[])
arr2 = ma.masked_array([])
print(f"With mask=[]: {arr1.mask}, type={type(arr1.mask)}, is_nomask={arr1.mask is ma.nomask}")
print(f"Without mask: {arr2.mask}, type={type(arr2.mask)}, is_nomask={arr2.mask is ma.nomask}")

# Test 2: Non-empty arrays behavior
print("\n2. Non-empty arrays with explicit masks:")
arr3 = ma.masked_array([1, 2], mask=[False, False])
arr4 = ma.masked_array([1, 2])
print(f"With mask=[F,F]: {arr3.mask}, type={type(arr3.mask)}, is_nomask={arr3.mask is ma.nomask}")
print(f"Without mask:    {arr4.mask}, type={type(arr4.mask)}, is_nomask={arr4.mask is ma.nomask}")

# Test 3: Concatenation preserving representation
print("\n3. Concatenation behavior:")
print("a) Two arrays with explicit empty masks:")
a1 = ma.masked_array([], mask=[])
a2 = ma.masked_array([], mask=[])
result = ma.concatenate([a1, a2])
print(f"  Input masks: {a1.mask} (array), {a2.mask} (array)")
print(f"  Result mask: {result.mask}, type={type(result.mask)}")

print("\nb) Two arrays without masks (nomask):")
b1 = ma.masked_array([])
b2 = ma.masked_array([])
result2 = ma.concatenate([b1, b2])
print(f"  Input masks: {b1.mask} (nomask={b1.mask is ma.nomask}), {b2.mask} (nomask={b2.mask is ma.nomask})")
print(f"  Result mask: {result2.mask}, type={type(result2.mask)}, is_nomask={result2.mask is ma.nomask}")

print("\nc) Mixed: one with array mask, one with nomask:")
c1 = ma.masked_array([1], mask=[False])
c2 = ma.masked_array([2])
result3 = ma.concatenate([c1, c2])
print(f"  Input masks: {c1.mask} (array), {c2.mask} (nomask={c2.mask is ma.nomask})")
print(f"  Result mask: {result3.mask}, type={type(result3.mask)}")

# Test 4: Check if this is really about optimization
print("\n4. Performance optimization check:")
large_unmasked = ma.masked_array(np.ones(1000))
print(f"Large unmasked array mask: is_nomask={large_unmasked.mask is ma.nomask}")

large_masked = ma.masked_array(np.ones(1000), mask=np.zeros(1000, dtype=bool))
print(f"Large array with explicit False mask: type={type(large_masked.mask)}, shape={large_masked.mask.shape}")