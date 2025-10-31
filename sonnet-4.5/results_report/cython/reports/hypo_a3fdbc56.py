from hypothesis import given, strategies as st
from Cython.Build.Dependencies import DistutilsInfo


@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5))
def test_distutils_info_merge_no_aliasing(libs):
    info1 = DistutilsInfo()
    info2 = DistutilsInfo()
    info2.values['libraries'] = libs[:]

    result = info1.merge(info2)

    assert result is info1, "merge should modify and return self"
    assert result.values['libraries'] is not info2.values['libraries'], \
        "Merged list should be a copy, not an alias"

    result.values['libraries'].append('new_lib')
    assert 'new_lib' not in info2.values['libraries'], \
        "Modifying merged list should not affect source"

if __name__ == "__main__":
    test_distutils_info_merge_no_aliasing()