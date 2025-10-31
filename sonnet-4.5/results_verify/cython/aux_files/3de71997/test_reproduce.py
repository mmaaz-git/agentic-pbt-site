from Cython.Tempita._looper import looper

print("Testing odd/even properties:")
for loop_obj, item in looper([1, 2, 3, 4]):
    pos = loop_obj.index
    print(f"Position {pos}: odd={loop_obj.odd}, even={loop_obj.even}")

print("\nAnalyzing the logic:")
for loop_obj, item in looper([1, 2, 3, 4]):
    pos = loop_obj.index
    expected_even = (pos % 2 == 0)
    expected_odd = (pos % 2 == 1)
    print(f"Position {pos}:")
    print(f"  Expected: even={expected_even}, odd={expected_odd}")
    print(f"  Actual:   even={loop_obj.even}, odd={loop_obj.odd}")
    print(f"  Match: even={loop_obj.even == expected_even}, odd={loop_obj.odd == expected_odd}")