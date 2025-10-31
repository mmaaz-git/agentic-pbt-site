import attr

class BuggyValidator:
    def __call__(self, inst, attr, value):
        # This should raise a NameError
        undefined_variable

@attr.define
class TestClass:
    value: int = attr.field(
        validator=attr.validators.or_(
            BuggyValidator(),
            attr.validators.instance_of(int)
        )
    )

# This should crash with NameError but doesn't
obj = TestClass(42)
print(f"Created object with value: {obj.value}")
print("The NameError from BuggyValidator was silently swallowed!")