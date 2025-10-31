import pandas as pd
import io

csv_data = "1,2,3\n4,5,6"
names = [[1, 2], [3, 4], [5, 6]]

try:
    df = pd.read_csv(io.StringIO(csv_data), names=names, header=None)
except TypeError as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")
    print("Expected: ValueError according to docstring")
except ValueError as e:
    print(f"Correct: Raised {type(e).__name__}: {e}")