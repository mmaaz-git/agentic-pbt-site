from pandas.io.formats.printing import adjoin


def custom_strlen(s):
    print(f"custom_strlen called with: '{s}'")
    return len(s) + 10


# Test case demonstrating the inconsistency
result = adjoin(1, ["a", "bb"], ["c", "dd"], strlen=custom_strlen)
print("\nResult:")
print(repr(result))
print("\nFormatted output:")
print(result)

print("\n" + "="*50)
print("Notice: custom_strlen was only called for the first list items ('a' and 'bb'),")
print("but NOT for the second list items ('c' and 'dd').")
print("This demonstrates the bug where strlen is not applied consistently to all lists.")