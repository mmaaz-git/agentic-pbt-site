import numpy as np
from scipy.odr import Data, ODR, unilinear
import tempfile

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
odr_obj = ODR(data, unilinear, beta0=[1.0, 0.0])

with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    odr_obj.rptfile = f.name

print("Calling set_iprint with init=3 (invalid value)...")
try:
    odr_obj.set_iprint(init=3)
    print("No error raised!")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Other error raised: {type(e).__name__}: {e}")

print("\n---Testing other invalid values---")
for val in [-1, 4, 10]:
    print(f"\nTesting init={val}...")
    odr_obj2 = ODR(data, unilinear, beta0=[1.0, 0.0])
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
        odr_obj2.rptfile = f2.name
    try:
        odr_obj2.set_iprint(init=val)
        print(f"  No error raised for init={val}!")
    except ValueError as e:
        print(f"  ValueError: {e}")
    except Exception as e:
        print(f"  Other error: {type(e).__name__}: {e}")

print("\n---Testing valid values (should work)---")
for val in [0, 1, 2]:
    print(f"\nTesting init={val}...")
    odr_obj3 = ODR(data, unilinear, beta0=[1.0, 0.0])
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f3:
        odr_obj3.rptfile = f3.name
    try:
        odr_obj3.set_iprint(init=val)
        print(f"  Success! init={val} worked correctly")
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")