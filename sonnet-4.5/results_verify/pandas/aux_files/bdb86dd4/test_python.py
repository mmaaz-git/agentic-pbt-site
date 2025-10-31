print("Testing regular Python behavior:")
print(f"1 ** -1 = {1 ** -1}")
print(f"1 ** -2 = {1 ** -2}")
print(f"1 ** -100 = {1 ** -100}")

print("\nTesting with integers explicitly:")
base = int(1)
exp = int(-1)
print(f"int(1) ** int(-1) = {base ** exp}")

print("\nFor comparison, 2 ** -1:")
print(f"2 ** -1 = {2 ** -1}")