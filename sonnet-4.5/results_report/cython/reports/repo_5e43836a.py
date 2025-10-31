from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

print("Testing with regular str:")
print(f"is_valid_tag('.0') = {is_valid_tag('.0')}")
print(f"is_valid_tag('.123') = {is_valid_tag('.123')}")
print(f"is_valid_tag('.999999') = {is_valid_tag('.999999')}")

print("\nTesting with EncodedString:")
print(f"is_valid_tag(EncodedString('.0')) = {is_valid_tag(EncodedString('.0'))}")
print(f"is_valid_tag(EncodedString('.123')) = {is_valid_tag(EncodedString('.123'))}")
print(f"is_valid_tag(EncodedString('.999999')) = {is_valid_tag(EncodedString('.999999'))}")

print("\nTesting valid identifiers:")
print(f"is_valid_tag('valid_name') = {is_valid_tag('valid_name')}")
print(f"is_valid_tag('.non_decimal') = {is_valid_tag('.non_decimal')}")