from scipy.odr import Data, Model, ODR
import numpy as np

def fcn(beta, x):
    return beta[0] * x + beta[1]

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
model = Model(fcn)

odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile='test.rpt')
odr_obj.set_iprint(init=0, so_init=1)