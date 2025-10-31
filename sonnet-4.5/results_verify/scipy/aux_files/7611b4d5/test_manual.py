import numpy as np
import scipy.odr as odr

x = np.array([1, 2, 3, 4, 5])
y = 2.5 * x

unilin_model = odr.unilinear
data = odr.Data(x, y)

print("Attempting to create ODR with beta0=[1.0]...")
try:
    odr_obj = odr.ODR(data, unilin_model, beta0=[1.0])
    print("No error raised - unexpected!")
except IndexError as e:
    print(f"IndexError raised as expected: {e}")
    import traceback
    traceback.print_exc()

print("\nNow testing with beta0=[1.0, 0.0] as workaround...")
odr_obj = odr.ODR(data, unilin_model, beta0=[1.0, 0.0])
output = odr_obj.run()
print(f"Result with 2 parameters: beta = {output.beta}")