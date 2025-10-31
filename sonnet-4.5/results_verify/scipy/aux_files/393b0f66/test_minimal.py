from scipy.odr import Data, Model, ODR
import numpy as np

def fcn(beta, x):
    return beta[0] * x + beta[1]

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
model = Model(fcn)

odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile='test.rpt')

print("Testing set_iprint with init=0, so_init=1")
try:
    odr_obj.set_iprint(init=0, so_init=1)
    print("Success - no error raised")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Other exception raised: {e}")