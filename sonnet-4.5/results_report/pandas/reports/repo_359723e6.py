from pandas.core.dtypes.common import ensure_python_int

# Test with float infinity - this should raise TypeError according to docs
# but actually raises OverflowError
ensure_python_int(float('inf'))