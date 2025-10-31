import numpy as np

a = np.array([1.0 + 2.0j, 3.0 + 4.0j])

print("Testing numpy.fft.rfft with complex input...")
print(f"Input array: {a}")
print(f"Input dtype: {a.dtype}")
print()

try:
    result = np.fft.rfft(a)
    print("Success:", result)
except TypeError as e:
    print("Bug confirmed - TypeError:", e)
    print()
    print("Documentation says (numpy/fft/_pocketfft.py:399):")
    print("  'If the input `a` contains an imaginary part,")
    print("   it is silently discarded.'")
    print()
    print("But rfft raises TypeError instead.")

print("\n" + "="*60 + "\n")
print("Now testing with just the real part:")
real_only = a.real
print(f"Real-only array: {real_only}")
print(f"Real-only dtype: {real_only.dtype}")

try:
    result_real = np.fft.rfft(real_only)
    print("Success with real-only input:", result_real)
except Exception as e:
    print("Error with real-only:", e)