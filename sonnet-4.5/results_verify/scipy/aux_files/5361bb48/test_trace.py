import traceback
from scipy import integrate

rn_list = [0, 0.5, 2]
print(f"Testing newton_cotes with list: {rn_list}")
print()

try:
    an, Bn = integrate.newton_cotes(rn_list, equal=0)
except TypeError as e:
    print("Full traceback:")
    traceback.print_exc()
    print()
    print(f"Error: {e}")