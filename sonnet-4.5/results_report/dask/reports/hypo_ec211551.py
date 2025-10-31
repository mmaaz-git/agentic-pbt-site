from hypothesis import given, settings, strategies as st
from dask.dataframe.dask_expr._util import LRU


@given(
    maxsize=st.integers(min_value=1, max_value=10),
    operations=st.lists(
        st.tuples(
            st.sampled_from(['set', 'get']),
            st.integers(min_value=0, max_value=20)
        ),
        min_size=0,
        max_size=100
    )
)
@settings(max_examples=500)
def test_lru_eviction_order(maxsize, operations):
    lru = LRU(maxsize)
    access_order = []

    for op_type, key in operations:
        if op_type == 'set':
            lru[key] = key
            if key in access_order:
                access_order.remove(key)
            access_order.append(key)
            if len(access_order) > maxsize:
                access_order.pop(0)
        elif op_type == 'get' and key in lru:
            _ = lru[key]
            access_order.remove(key)
            access_order.append(key)

    assert set(lru.keys()) == set(access_order), f"LRU keys {set(lru.keys())} != expected {set(access_order)}"


if __name__ == "__main__":
    test_lru_eviction_order()