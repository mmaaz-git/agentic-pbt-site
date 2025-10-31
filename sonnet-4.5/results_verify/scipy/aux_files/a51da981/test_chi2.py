import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy.stats import chi2

print("Testing chi2.sf(0, df=2) - survival function at 0:")
result = chi2.sf(0, df=2)
print(f"chi2.sf(0, df=2) = {result}")
print(f"Expected: 1.0")
print(f"Match: {result == 1.0}")

print("\nAdditional tests:")
print(f"chi2.sf(0, df=1) = {chi2.sf(0, df=1)}")
print(f"chi2.sf(0, df=3) = {chi2.sf(0, df=3)}")
print(f"chi2.sf(0, df=10) = {chi2.sf(0, df=10)}")

print("\nTesting small positive values:")
print(f"chi2.sf(1e-10, df=2) = {chi2.sf(1e-10, df=2)}")
print(f"chi2.sf(1e-300, df=2) = {chi2.sf(1e-300, df=2)}")