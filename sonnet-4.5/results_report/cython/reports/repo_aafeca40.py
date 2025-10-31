from Cython.Tempita._looper import looper

print("Testing looper odd/even properties:")
print("=" * 50)

for loop_obj, item in looper([1, 2, 3, 4]):
    pos = loop_obj.index
    print(f"Position {pos}: odd={loop_obj.odd}, even={loop_obj.even}")

    # Show what the values SHOULD be based on mathematical definitions
    expected_odd = pos % 2 == 1
    expected_even = pos % 2 == 0
    print(f"  Expected: odd={expected_odd}, even={expected_even}")

    # Check if they match
    if loop_obj.odd != expected_odd or bool(loop_obj.even) != expected_even:
        print(f"  ❌ MISMATCH!")
    else:
        print(f"  ✓ Match")
    print()

print("=" * 50)
print("\nDetailed analysis of position 0:")
loop_list = list(looper([1]))
loop_obj, item = loop_list[0]
print(f"Position: {loop_obj.index}")
print(f"odd property returns: {loop_obj.odd} (type: {type(loop_obj.odd).__name__})")
print(f"even property returns: {loop_obj.even} (type: {type(loop_obj.even).__name__})")
print(f"\nMathematically, 0 % 2 = {0 % 2}, so position 0 is EVEN")
print(f"Therefore, odd should be False and even should be True")
print(f"But we got odd={loop_obj.odd} and even={loop_obj.even}")