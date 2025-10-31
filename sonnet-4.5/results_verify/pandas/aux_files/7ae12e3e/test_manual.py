from pandas.io.formats.printing import adjoin


def custom_strlen(s):
    print(f"custom_strlen called with: {repr(s)}")
    return len(s) + 10


result = adjoin(1, ["a"], ["b"], strlen=custom_strlen)
print("Result:")
print(repr(result))

print("\n\nNow let's also test with multiple strings per list:")
result2 = adjoin(1, ["a", "bb"], ["c", "dd"], strlen=custom_strlen)
print("Result2:")
print(repr(result2))