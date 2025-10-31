import numpy.rec

lst = [float('nan'), float('nan'), 1.0, 1.0]
result = numpy.rec.find_duplicate(lst)

print(f"Input: {lst}")
print(f"Result: {result}")
print(f"Result length: {len(result)}")
print(f"Expected length: 2 (both NaN and 1.0 should be in the result)")

assert len(result) == 2, f"Expected 2 duplicates (NaN and 1.0), but got {len(result)}"