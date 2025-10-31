from scipy.odr import Data, Model, ODR
import numpy as np

def fcn(beta, x):
    return beta[0] * x + beta[1]

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
model = Model(fcn)

# Test all possible combinations of init and so_init
for init in [0, 1, 2]:
    for so_init in [0, 1, 2]:
        odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile='test.rpt')
        try:
            odr_obj.set_iprint(init=init, so_init=so_init)
            print(f"✓ init={init}, so_init={so_init}: Success")
        except ValueError as e:
            print(f"✗ init={init}, so_init={so_init}: Failed - {e}")
        except Exception as e:
            print(f"✗ init={init}, so_init={so_init}: Failed - {type(e).__name__}: {e}")