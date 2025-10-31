import numpy as np

# Test the mathematical operation that's failing
# From the error, it seems acosh is getting an invalid value

def test_acosh_domain():
    """Test the domain of acosh function"""
    print("Testing acosh domain requirements:")
    print("acosh(x) is only defined for x >= 1\n")

    # Test values around the boundary
    test_values = [0.5, 0.9, 1.0, 1.1, 2.0, -1.0, -10.0]

    for val in test_values:
        try:
            result = np.arccosh(val)
            print(f"acosh({val:6.1f}) = {result:10.6f}")
        except:
            # Try anyway to see what happens
            with np.errstate(invalid='ignore'):
                result = np.arccosh(val)
                print(f"acosh({val:6.1f}) = {result} (warning suppressed)")

test_acosh_domain()

print("\n" + "="*50)
print("Testing the Taylor window B calculation:")
print("="*50)

# The Taylor window uses: B = 10**(sll/20)
# Then computes: A = acosh(B) / pi

sll_values = [-10.0, 0.0, 10.0, 20.0, 30.0]

for sll in sll_values:
    B = 10**(sll/20)
    print(f"sll = {sll:6.1f}: B = 10^({sll}/20) = {B:12.8f}")
    print(f"  B >= 1? {B >= 1}")
    if B >= 1:
        A = np.arccosh(B) / np.pi
        print(f"  A = acosh(B)/π = {A:12.8f}")
    else:
        print(f"  acosh(B) is undefined since B < 1")
    print()

print("="*50)
print("Conclusion:")
print("="*50)
print("The Taylor window algorithm requires B = 10^(sll/20) >= 1")
print("This means sll/20 >= 0, so sll >= 0")
print("Negative sll values lead to B < 1, causing acosh to return NaN")