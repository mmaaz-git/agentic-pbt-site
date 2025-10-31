import numpy as np

# Test the logic that's in the numpy code
def test_logic():
    # Simulate the logic from lines 162-167
    order = 'C'  # This is always 'C' or 'F'

    # Test with contiguous array
    arr_contiguous = np.array([[1, 2], [3, 4]])
    print(f"Contiguous array flags.contiguous: {arr_contiguous.flags.contiguous}")
    result1 = not (order or arr_contiguous.flags.contiguous)
    print(f"not (order or arr.flags.contiguous) = not ('{order}' or {arr_contiguous.flags.contiguous}) = {result1}")

    # Test with non-contiguous array
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    arr_noncontiguous = arr[::2, ::2]
    print(f"\nNon-contiguous array flags.contiguous: {arr_noncontiguous.flags.contiguous}")
    result2 = not (order or arr_noncontiguous.flags.contiguous)
    print(f"not (order or arr.flags.contiguous) = not ('{order}' or {arr_noncontiguous.flags.contiguous}) = {result2}")

    # Show what the intended logic probably was
    print(f"\nIntended logic (not arr.flags.contiguous) for non-contiguous: {not arr_noncontiguous.flags.contiguous}")

    # Test truthiness of 'C' and 'F'
    print(f"\nbool('C') = {bool('C')}")
    print(f"bool('F') = {bool('F')}")
    print(f"'C' or False = {'C' or False}")
    print(f"'F' or False = {'F' or False}")

test_logic()