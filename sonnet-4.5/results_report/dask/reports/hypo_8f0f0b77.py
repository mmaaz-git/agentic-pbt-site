import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from dask.dataframe.dask_expr.io.parquet import _aggregate_statistics_to_file

stat_dict = st.fixed_dictionaries({
    "num_rows": st.integers(min_value=0, max_value=10000),
    "num_row_groups": st.integers(min_value=1, max_value=10),
    "serialized_size": st.integers(min_value=0, max_value=1000000),
    "row_groups": st.lists(
        st.fixed_dictionaries({
            "num_rows": st.integers(min_value=0, max_value=1000),
            "total_byte_size": st.integers(min_value=0, max_value=100000),
            "columns": st.lists(
                st.fixed_dictionaries({
                    "total_compressed_size": st.integers(min_value=0, max_value=10000),
                    "total_uncompressed_size": st.integers(min_value=0, max_value=10000),
                    "path_in_schema": st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L',))),
                    "statistics": st.one_of(
                        st.none(),
                        st.fixed_dictionaries({
                            "min": st.integers(),
                            "max": st.integers(),
                            "null_count": st.integers(min_value=0, max_value=1000),
                            "num_values": st.integers(min_value=0, max_value=1000),
                            "distinct_count": st.integers(min_value=0, max_value=1000),
                        })
                    )
                }),
                min_size=1,
                max_size=5
            )
        }),
        min_size=1,
        max_size=5
    )
})

@given(st.lists(stat_dict, min_size=1, max_size=5))
@settings(max_examples=100)
def test_aggregate_statistics_preserves_num_files(stats):
    result = _aggregate_statistics_to_file(stats)
    assert len(result) == len(stats)

if __name__ == "__main__":
    test_aggregate_statistics_preserves_num_files()