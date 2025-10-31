import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6]})
rolling = df.rolling(window=2, step=0)
result = rolling.mean()
print(result)