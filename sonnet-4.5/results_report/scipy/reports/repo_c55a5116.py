from scipy.integrate import tanhsinh

def f(x):
    return 1.0

result = tanhsinh(f, 0.0, 1.0)
print(f"Integral result: {result.integral}")