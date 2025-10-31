import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy import special

a, b, y = 1.0, 0.1, 0.01

x = special.betainccinv(a, b, y)
print(f"betainccinv({a}, {b}, {y}) = {x}")

result = special.betaincc(a, b, x)
print(f"betaincc({a}, {b}, {x}) = {result}")
print(f"Expected: {y}")
print(f"Error: {abs(result - y)}")

# Let's also check what betaincc(1.0, 0.1, 1.0) actually returns
print(f"\nVerifying: betaincc({a}, {b}, 1.0) = {special.betaincc(a, b, 1.0)}")

# And let's check what betainccinv should return for y=0
print(f"betainccinv({a}, {b}, 0.0) = {special.betainccinv(a, b, 0.0)}")
print(f"betaincc({a}, {b}, 1.0) = {special.betaincc(a, b, 1.0)}")