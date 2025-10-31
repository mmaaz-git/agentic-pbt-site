from scipy.special import boxcox1p, inv_boxcox1p

y = 1.0
lmbda = 5.808166112732823e-234

x = inv_boxcox1p(y, lmbda)
y_recovered = boxcox1p(x, lmbda)

print(f"Input: y={y}, lambda={lmbda}")
print(f"inv_boxcox1p(y, lambda) = {x}")
print(f"boxcox1p(x, lambda) = {y_recovered}")
print(f"Expected: {y}")
print(f"Actual: {y_recovered}")
print(f"Error: {abs(y_recovered - y)}")