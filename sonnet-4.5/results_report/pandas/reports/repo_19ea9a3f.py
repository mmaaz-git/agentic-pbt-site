from pandas.util._validators import _check_for_default_values


class UncomparableObject:
    def __eq__(self, other):
        raise TypeError("Cannot compare UncomparableObject")


obj = UncomparableObject()
arg_val_dict = {'param': obj}
compat_args = {'param': obj}

_check_for_default_values('test_func', arg_val_dict, compat_args)