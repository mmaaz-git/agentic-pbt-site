from scipy.constants import precision, physical_constants

key = 'electron to shielded proton magn. moment ratio'
prec = precision(key)
print(f"precision('{key}') = {prec}")

value, unit, uncertainty = physical_constants[key]
print(f"Value: {value}")
print(f"Unit: {unit}")
print(f"Uncertainty: {uncertainty}")
print(f"Calculated precision (uncertainty/value): {uncertainty / value}")
print(f"Expected (non-negative): {abs(uncertainty / value)}")

print("\nAdditional examples of negative precision values:")
negative_precision_keys = [
    'electron-deuteron magn. moment ratio',
    'muon magn. moment',
    'electron charge to mass quotient'
]

for k in negative_precision_keys:
    try:
        p = precision(k)
        v = physical_constants[k][0]
        print(f"  {k}: value={v:.6e}, precision={p:.6e}")
    except:
        pass