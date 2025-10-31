import numpy.rec

lst = [float('nan'), float('nan'), 1.0, 1.0]
result = numpy.rec.find_duplicate(lst)

print(f"Input: {lst}")
print(f"Result: {result}")

print(f"\nExpected result length: 2 (both nan and 1.0 are duplicated)")
print(f"Actual result length: {len(result)}")

if len(result) != 2:
    print("BUG CONFIRMED: Result should contain 2 elements (nan and 1.0), but only contains 1")