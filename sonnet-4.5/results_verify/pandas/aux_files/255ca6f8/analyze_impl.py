import numpy as np
import numpy.ma as ma

def trace_allequal(a, b, fill_value=True):
    """Trace the execution path of allequal"""
    print(f"=== allequal(a, b, fill_value={fill_value}) ===")

    # Get masks
    m = ma.mask_or(ma.getmask(a), ma.getmask(b))
    print(f"Combined mask m: {m}")

    if m is ma.nomask:
        print("No masks - simple comparison")
        x = ma.getdata(a)
        y = ma.getdata(b)
        d = np.equal(x, y)
        result = d.all()
        print(f"  Result: {result}")
        return result
    elif fill_value:
        print(f"Has masks, fill_value=True")
        x = ma.getdata(a)
        y = ma.getdata(b)
        print(f"  x data: {x}")
        print(f"  y data: {y}")
        d = np.equal(x, y)
        print(f"  Element-wise equal: {d}")
        dm = ma.array(d, mask=m, copy=False)
        print(f"  Masked array dm: {dm}")
        filled = dm.filled(True)
        print(f"  Filled with True: {filled}")
        result = filled.all(None)
        print(f"  Result: {result}")
        return result
    else:
        print(f"Has masks, fill_value=False - ALWAYS RETURNS False")
        return False

# Test case: identical arrays with same mask
print("TEST 1: Identical arrays with identical masks")
x = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y = ma.array([1.0, 999.0, 3.0], mask=[False, True, False])
print(f"x: {x}")
print(f"y: {y}")

result_true = trace_allequal(x, y, fill_value=True)
result_false = trace_allequal(x, y, fill_value=False)

print(f"\nActual allequal(x, y, fill_value=True): {ma.allequal(x, y, fill_value=True)}")
print(f"Actual allequal(x, y, fill_value=False): {ma.allequal(x, y, fill_value=False)}")

print("\n" + "="*60)
print("TEST 2: Different values at unmasked positions")
x2 = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y2 = ma.array([1.0, 999.0, 5.0], mask=[False, True, False])
print(f"x2: {x2}")
print(f"y2: {y2}")

result2_true = trace_allequal(x2, y2, fill_value=True)
result2_false = trace_allequal(x2, y2, fill_value=False)

print(f"\nActual allequal(x2, y2, fill_value=True): {ma.allequal(x2, y2, fill_value=True)}")
print(f"Actual allequal(x2, y2, fill_value=False): {ma.allequal(x2, y2, fill_value=False)}")