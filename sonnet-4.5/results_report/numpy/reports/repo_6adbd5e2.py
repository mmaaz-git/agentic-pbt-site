import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

from numpy.f2py.symbolic import as_symbol, as_integer

# Create expressions
a = as_symbol('a')
b = as_integer(0, 8)  # 0 with kind=8
c = as_integer(1, 4)  # 1 with kind=4

# Test associativity: (a + b) + c vs a + (b + c)
left_assoc = (a + b) + c
right_assoc = a + (b + c)

print("Testing addition associativity violation:")
print(f"a = {a}")
print(f"b = {b} (0 with kind=8)")
print(f"c = {c} (1 with kind=4)")
print()
print(f"(a + 0_8) + 1_4 = {left_assoc}")
print(f"a + (0_8 + 1_4) = {right_assoc}")
print()
print(f"Are they equal? {left_assoc == right_assoc}")
print(f"Should be equal for associativity to hold: True")
print()
print("This violates the fundamental property of associativity: (a + b) + c == a + (b + c)")