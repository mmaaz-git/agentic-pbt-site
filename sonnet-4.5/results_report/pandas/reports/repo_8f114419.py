from pandas.io.sas import read_sas

filename = "data.xpt.backup"
try:
    read_sas(filename)
    print(f"BUG: '{filename}' was detected as xport format (FileNotFoundError raised)")
    print("Expected: ValueError('unable to infer format...')")
except FileNotFoundError as e:
    print(f"BUG: '{filename}' was detected as xport format (FileNotFoundError raised)")
    print(f"Error: {e}")
    print("Expected: ValueError('unable to infer format...')")
except ValueError as e:
    if "unable to infer format" in str(e):
        print(f"CORRECT: '{filename}' was not detected as a SAS file")
        print(f"ValueError raised: {e}")
    else:
        print(f"Different ValueError: {e}")