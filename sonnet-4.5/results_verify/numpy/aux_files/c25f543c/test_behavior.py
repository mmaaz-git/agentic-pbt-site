import numpy as np
import numpy.ma as ma

def analyze_behavior(x, y, desc):
    """Analyze the behavior of allequal with different fill_values"""
    print(f"\n{desc}")
    print(f"  x: data={x.data}, mask={x.mask}")
    print(f"  y: data={y.data}, mask={y.mask}")

    # Get the combined mask
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    print(f"  Combined mask: {m}")

    if m is ma.nomask:
        print(f"  No mask case - comparing all values")
    else:
        print(f"  Has mask - some values are masked")
        # Get unmasked values
        unmasked_indices = ~m
        print(f"  Unmasked indices: {unmasked_indices}")
        if unmasked_indices.any():
            print(f"  Unmasked x values: {x.data[unmasked_indices]}")
            print(f"  Unmasked y values: {y.data[unmasked_indices]}")
            print(f"  Are unmasked values equal? {np.array_equal(x.data[unmasked_indices], y.data[unmasked_indices])}")

    result_true = ma.allequal(x, y, fill_value=True)
    result_false = ma.allequal(x, y, fill_value=False)

    print(f"  Result with fill_value=True: {result_true}")
    print(f"  Result with fill_value=False: {result_false}")

    return result_false

# Test case 1: Identical arrays with some masked values (same mask)
x1 = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y1 = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
analyze_behavior(x1, y1, "Case 1: Identical arrays with same mask")

# Test case 2: Same unmasked values, different masked values
x2 = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y2 = ma.array([1.0, 999.0, 3.0], mask=[False, True, False])
analyze_behavior(x2, y2, "Case 2: Same unmasked values, different masked values")

# Test case 3: Different masks but same unmasked values
x3 = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y3 = ma.array([1.0, 2.0, 3.0], mask=[False, False, True])
analyze_behavior(x3, y3, "Case 3: Different masks, but values match where both unmasked")

# Test case 4: The documentation example
x4 = ma.array([1e10, 1e-7, 42.0], mask=[0, 0, 1])
y4 = np.array([1e10, 1e-7, -42.0])
analyze_behavior(ma.asarray(x4), ma.asarray(y4), "Case 4: Documentation example")