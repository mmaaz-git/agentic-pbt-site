from pandas.io.sas import read_sas

filename = "data.xpt.backup"
try:
    read_sas(filename)
except FileNotFoundError:
    print(f"BUG: '{filename}' was detected as xport format")
    print("Expected: ValueError('unable to infer format...')")
except ValueError as e:
    print(f"Correct: {e}")