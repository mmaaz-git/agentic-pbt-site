import pandas as pd

print("Testing NA propagation in pandas masked arrays")
print("="*50)

na_float = pd.array([None], dtype="Float64")
na_int = pd.array([None], dtype="Int64")

print("Float64 Array:")
print(f"NA ** 0 = {(na_float ** 0)[0]}")
print(f"NA * 0 = {(na_float * 0)[0]}")
print()

print("Int64 Array:")
print(f"Integer: NA ** 0 = {(na_int ** 0)[0]}")
print(f"Integer: NA * 0 = {(na_int * 0)[0]}")
print()

# Test other operations with identity elements
print("Testing other operations with identity elements:")
print(f"NA + 0 = {(na_float + 0)[0]}")
print(f"NA - 0 = {(na_float - 0)[0]}")
print(f"NA * 1 = {(na_float * 1)[0]}")
print(f"NA / 1 = {(na_float / 1)[0]}")
print(f"NA ** 1 = {(na_float ** 1)[0]}")
print()

# Test asymmetric behavior
print("Testing asymmetric behavior:")
zero_array = pd.array([0], dtype="Float64")
print(f"0 ** NA = {(zero_array ** na_float)[0]}")