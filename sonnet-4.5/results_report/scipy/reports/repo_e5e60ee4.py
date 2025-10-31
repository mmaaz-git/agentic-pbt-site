import numpy as np
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder

# Create data with exact values from the bug report
data = np.array([703_036_036_854_775_809, -8_520_336_000_000_000_000], dtype='datetime64[ns]')
original_var = Variable(('time',), data)

# Create CFDatetimeCoder with use_cftime=False
coder = CFDatetimeCoder(use_cftime=False, time_unit='ns')

# Attempt to encode, which should trigger the AttributeError
try:
    encoded_var = coder.encode(original_var)
    print("Successfully encoded!")
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"Unexpected error ({type(e).__name__}): {e}")