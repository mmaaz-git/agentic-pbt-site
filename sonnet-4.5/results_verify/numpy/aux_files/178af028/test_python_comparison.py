import numpy as np
import numpy.strings as nps

print("Comparing numpy.strings.replace to Python str.replace")
print("="*60)

test_cases = [
    (['0'], '0', '00', 1),
    (['hello'], 'l', 'LL', 2),
    (['test'], 't', '', 1),
    (['a', 'aa', 'aaa'], 'a', 'bb', 1),
    (['x'], 'x', 'xyz', -1),
    (['cat'], 'cat', 'elephant', 1)
]

for arr_list, old, new, count in test_cases:
    print(f"\nTest: arr={arr_list}, old='{old}', new='{new}', count={count}")
    print("-"*50)

    # Test with minimal dtype
    arr_minimal = np.array(arr_list)
    print(f"Minimal dtype: {arr_minimal.dtype}")

    result_minimal = nps.replace(arr_minimal, old, new, count=count)
    print(f"Result dtype: {result_minimal.dtype}")

    all_match = True
    for i in range(len(arr_list)):
        expected = arr_list[i].replace(old, new, count)
        actual = str(result_minimal[i])
        match = actual == expected
        if not match:
            all_match = False
        print(f"  '{arr_list[i]}' -> NumPy: '{actual}', Python: '{expected}', Match: {match}")

    print(f"Overall match with Python semantics: {all_match}")

    # If it doesn't match, try with a larger dtype
    if not all_match:
        # Calculate needed size
        max_result_len = max(len(s.replace(old, new, count)) for s in arr_list)
        arr_sized = np.array(arr_list, dtype=f'U{max_result_len}')
        print(f"\nRetrying with dtype U{max_result_len}:")
        result_sized = nps.replace(arr_sized, old, new, count=count)
        for i in range(len(arr_list)):
            expected = arr_list[i].replace(old, new, count)
            actual = str(result_sized[i])
            print(f"  '{arr_list[i]}' -> NumPy: '{actual}', Match: {actual == expected}")