import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a single-row DataFrame with specific numeric values
df = pd.DataFrame({
    'A': [0.0],
    'B': [-1.297501e+16],
    'C': [-1.297501e+16]
})

# This should crash with ValueError
pd.plotting.scatter_matrix(df)