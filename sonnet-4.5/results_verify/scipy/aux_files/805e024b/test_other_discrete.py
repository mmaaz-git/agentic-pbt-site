import scipy.stats as stats
import numpy as np

# Test how other discrete distributions handle ppf(1.0)

# 1. Binomial distribution
print("Binomial distribution (n=10, p=0.5):")
for k in [8, 9, 10]:
    cdf_k = stats.binom.cdf(k, n=10, p=0.5)
    ppf_result = stats.binom.ppf(cdf_k, n=10, p=0.5)
    print(f"  k={k}: cdf={cdf_k:.10f}, ppf(cdf)={ppf_result}")
ppf_1 = stats.binom.ppf(1.0, n=10, p=0.5)
print(f"  ppf(1.0) = {ppf_1}")

print("\n" + "="*50)

# 2. Geometric distribution
print("Geometric distribution (p=0.3):")
for k in [15, 20, 25]:
    cdf_k = stats.geom.cdf(k, p=0.3)
    ppf_result = stats.geom.ppf(cdf_k, p=0.3)
    print(f"  k={k}: cdf={cdf_k:.10f}, ppf(cdf)={ppf_result}")
ppf_1 = stats.geom.ppf(1.0, p=0.3)
print(f"  ppf(1.0) = {ppf_1}")

print("\n" + "="*50)

# 3. Negative binomial distribution
print("Negative binomial distribution (n=5, p=0.3):")
for k in [15, 20, 25]:
    cdf_k = stats.nbinom.cdf(k, n=5, p=0.3)
    ppf_result = stats.nbinom.ppf(cdf_k, n=5, p=0.3)
    print(f"  k={k}: cdf={cdf_k:.10f}, ppf(cdf)={ppf_result}")
ppf_1 = stats.nbinom.ppf(1.0, n=5, p=0.3)
print(f"  ppf(1.0) = {ppf_1}")

print("\n" + "="*50)

# 4. Hypergeometric distribution (finite support)
print("Hypergeometric distribution (M=20, n=10, N=5):")
for k in [3, 4, 5]:
    cdf_k = stats.hypergeom.cdf(k, M=20, n=10, N=5)
    ppf_result = stats.hypergeom.ppf(cdf_k, M=20, n=10, N=5)
    print(f"  k={k}: cdf={cdf_k:.10f}, ppf(cdf)={ppf_result}")
ppf_1 = stats.hypergeom.ppf(1.0, M=20, n=10, N=5)
print(f"  ppf(1.0) = {ppf_1}")

print("\n" + "="*50)

# Test Poisson with different mu values
print("Poisson distribution with various mu values:")
for mu in [0.5, 1.0, 2.0, 10.0]:
    ppf_1 = stats.poisson.ppf(1.0, mu)
    print(f"  mu={mu:4.1f}: ppf(1.0) = {ppf_1}")