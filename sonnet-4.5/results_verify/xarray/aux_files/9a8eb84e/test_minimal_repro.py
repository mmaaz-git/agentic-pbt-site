import numpy as np
from datetime import datetime
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder

dt = datetime(2003, 1, 1, 0, 0, 0, 1)
datetime_arr = np.array([dt], dtype="datetime64[ns]")

encoding = {"units": "days since 2000-01-01", "calendar": "proleptic_gregorian"}
original_var = Variable(("time",), datetime_arr, encoding=encoding)

coder = CFDatetimeCoder()
encoded_var = coder.encode(original_var)
decoded_var = coder.decode(encoded_var)

print(f"Original: {original_var.data[0]} ({original_var.data.view('int64')[0]} ns)")
print(f"Decoded:  {decoded_var.data[0]} ({decoded_var.data.view('int64')[0]} ns)")
print(f"Lost precision: {original_var.data.view('int64')[0] - decoded_var.data.view('int64')[0]} ns")

assert np.array_equal(original_var.data, decoded_var.data), "Round-trip failed!"