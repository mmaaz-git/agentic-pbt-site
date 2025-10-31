from pandas.io.json._normalize import convert_to_line_delimits

# Test case 1: String that starts with [ but doesn't end with ]
s1 = "[abc"
result1 = convert_to_line_delimits(s1)
print(f"Input:  {repr(s1)}")
print(f"Output: {repr(result1)}")
print(f"Expected: {repr(s1)} (unchanged)")
print()

# Test case 2: Just a single [ character
s2 = "["
result2 = convert_to_line_delimits(s2)
print(f"Input:  {repr(s2)}")
print(f"Output: {repr(result2)}")
print(f"Expected: {repr(s2)} (unchanged)")
print()

# Test case 3: Valid JSON list
s3 = "[1,2,3]"
result3 = convert_to_line_delimits(s3)
print(f"Input:  {repr(s3)}")
print(f"Output: {repr(result3)}")
print(f"Expected: Line-delimited output")
print()

# Test case 4: String ending with ] but not starting with [
s4 = "abc]"
result4 = convert_to_line_delimits(s4)
print(f"Input:  {repr(s4)}")
print(f"Output: {repr(result4)}")
print(f"Expected: {repr(s4)} (unchanged)")