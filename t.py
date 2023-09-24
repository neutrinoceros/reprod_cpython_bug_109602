import numpy as np

from min_yt import StreamFieldInfo, StreamHierarchy, load_uniform_grid


def foo():
    shape = (16, 16, 16)
    ds = load_uniform_grid(
        domain_dimensions=shape
    )
    ds.index = StreamHierarchy(ds, dataset_type="test")

    # inline ds.create_field_info()
    ds.field_dependencies = {}
    ds.derived_field_list = []
    ds.field_info = StreamFieldInfo(ds, ds.index.field_list)
    ds.coordinates.setup_fields(ds.field_info)
    ds.field_info.setup_fluid_fields()
    ds.field_info.setup_fluid_index_fields()

    ds.field_info.load_all_plugins(ds.default_fluid_type)
    deps, unloaded = ds.field_info.check_derived_fields()


NLOOPS = 800
for i in range(1, NLOOPS + 1):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
