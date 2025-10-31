import pandas as pd

values = [2.2250738585e-313, -1.0]
result = pd.cut(values, bins=2)
print(result)