import scipy.fft._pocketfft.helper as _helper

# Test the underlying good_size function
for val in [-1, 0, 1]:
    try:
        result = _helper.good_size(val, True)
        print(f"_helper.good_size({val}, True) = {result}")
    except Exception as e:
        print(f"_helper.good_size({val}, True) raises: {type(e).__name__}: {e}")