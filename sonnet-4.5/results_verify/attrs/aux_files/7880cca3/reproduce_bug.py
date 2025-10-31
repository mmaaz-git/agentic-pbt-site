import attr
import inspect

print("gt validator docstring:")
print(inspect.getsource(attr.validators.gt))

validator = attr.validators.gt(10)
print(f"\nActual compare_func: {validator.compare_func}")

import operator
print(f"Is operator.gt? {validator.compare_func is operator.gt}")
print(f"Is operator.ge? {validator.compare_func is operator.ge}")

@attr.define
class TestClass:
    value: int = attr.field(validator=attr.validators.gt(10))

try:
    TestClass(10)
    print("\n10 passed validation (BUG if this happens)")
except ValueError:
    print("\n10 failed validation (correct behavior for >)")

try:
    TestClass(11)
    print("11 passed validation (correct behavior for >)")
except ValueError:
    print("11 failed validation (BUG if this happens)")