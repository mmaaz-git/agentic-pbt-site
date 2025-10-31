import scipy.fftpack
import inspect

# Get the docstring
print("=" * 60)
print("scipy.fftpack.next_fast_len docstring:")
print("=" * 60)
print(scipy.fftpack.next_fast_len.__doc__)

# Get the signature
print("\n" + "=" * 60)
print("Function signature:")
print("=" * 60)
print(inspect.signature(scipy.fftpack.next_fast_len))