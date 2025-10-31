from Cython.Utility import pylong_join

print("Testing with count=1, join_type='':")
result = pylong_join(1, join_type='')
print(f"Result: {result}")
print()

print("Testing with count=2, join_type='':")
result = pylong_join(2, join_type='')
print(f"Result: {result}")
print()

print("Checking for invalid empty casts '()':")
result1 = pylong_join(1, join_type='')
result2 = pylong_join(2, join_type='')
print(f"Result 1 contains '()': {('()' in result1)}")
print(f"Result 2 contains '()': {('()' in result2)}")