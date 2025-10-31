import numpy.rec

test_cases = [
    '\x00',
    'a\x00',
    '\x00b',
    'a\x00b',
]

print("Testing numpy.rec.fromrecords with null characters:")
print("-" * 60)
for s in test_cases:
    r = numpy.rec.fromrecords([(s,)], names='text')
    preserved = r.text[0] == s
    print(f"Input: {repr(s):10} → Output: {repr(str(r.text[0])):10} | Preserved: {preserved}")

print("\n" + "=" * 60 + "\n")

# Let's also test with numpy.array directly to confirm the claim
import numpy as np
print("Testing numpy.array with null characters (to verify it's a core issue):")
print("-" * 60)
for s in test_cases:
    arr = np.array([s])
    preserved = arr[0] == s
    print(f"Input: {repr(s):10} → Output: {repr(str(arr[0])):10} | Preserved: {preserved}")

print("\n" + "=" * 60 + "\n")

# Let's debug this further by examining the actual data
print("Deeper analysis - examining the dtype and actual storage:")
print("-" * 60)
for s in test_cases:
    r = numpy.rec.fromrecords([(s,)], names='text')
    arr = np.array([s])
    print(f"Input: {repr(s):10}")
    print(f"  rec.fromrecords result: {repr(r.text[0]):10} (dtype: {r.dtype['text']})")
    print(f"  numpy.array result:     {repr(arr[0]):10} (dtype: {arr.dtype})")
    print(f"  Length comparison: input={len(s)}, rec={len(r.text[0])}, arr={len(arr[0])}")
    print()