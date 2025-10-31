import numpy as np
from pandas.core.array_algos.putmask import putmask_without_repeat
from pandas.core.dtypes.common import is_list_like

def trace_putmask_without_repeat(values, mask, new):
    """Trace through the function logic"""
    print("=== Tracing putmask_without_repeat ===")
    print(f"values shape: {values.shape}, dtype: {values.dtype}")
    print(f"mask shape: {mask.shape}, sum: {mask.sum()}")
    print(f"new: {new}, shape: {new.shape if hasattr(new, 'shape') else 'N/A'}")

    # Line 75-76
    if getattr(new, "ndim", 0) >= 1:
        print(f"new.ndim = {new.ndim} >= 1, converting to values.dtype")
        new = new.astype(values.dtype, copy=False)

    # Line 79
    nlocs = mask.sum()
    print(f"nlocs (mask.sum()): {nlocs}")

    # Line 80
    if nlocs > 0 and is_list_like(new) and getattr(new, "ndim", 1) == 1:
        print(f"Entering main branch: nlocs={nlocs} > 0, is_list_like={is_list_like(new)}, ndim={new.ndim}")

        shape = np.shape(new)
        print(f"shape of new: {shape}")

        # Line 84 check
        if nlocs == shape[-1]:
            print(f"Branch 1: nlocs ({nlocs}) == shape[-1] ({shape[-1]})")
            print("Would call np.place(values, mask, new)")
        # Line 93 check - THIS IS THE BUG!
        elif mask.shape[-1] == shape[-1] or shape[-1] == 1:
            print(f"Branch 2: mask.shape[-1] ({mask.shape[-1]}) == shape[-1] ({shape[-1]}) or shape[-1] ({shape[-1]}) == 1")
            print(f"Condition breakdown:")
            print(f"  - mask.shape[-1] == shape[-1]: {mask.shape[-1]} == {shape[-1]} = {mask.shape[-1] == shape[-1]}")
            print(f"  - shape[-1] == 1: {shape[-1]} == 1 = {shape[-1] == 1}")
            print("Would call np.putmask(values, mask, new) - THIS WILL REPEAT!")
        else:
            print(f"Branch 3: Would raise ValueError")
    else:
        print("Would fall through to np.putmask at line 98")

# Test case from bug report
values = np.arange(10)
mask = np.ones(10, dtype=bool)
new = np.array([999])

trace_putmask_without_repeat(values.copy(), mask, new)

print("\n=== Actual execution ===")
values_copy = values.copy()
putmask_without_repeat(values_copy, mask, new)
print(f"Result: {values_copy}")