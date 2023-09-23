import numpy as np

from min_yt import load_uniform_grid


def foo():
    shape = (16, 16, 16)
    ds = load_uniform_grid(
        data={("gas", "density"): np.ones(shape)},
        domain_dimensions=shape,
    )
    ds._instantiated_index = ds._index_class(
        ds, dataset_type=ds.dataset_type
    )

    # inline ds.create_field_info()
    ds.field_dependencies = {}
    ds.derived_field_list = []
    ds.field_info = ds._field_info_class(ds, ds.field_list)
    ds.coordinates.setup_fields(ds.field_info)
    ds.field_info.setup_fluid_fields()
    ds.field_info.setup_fluid_index_fields()

    ds.field_info.load_all_plugins(ds.default_fluid_type)
    deps, unloaded = ds.field_info.check_derived_fields()
    ds.field_dependencies.update(deps)

NLOOPS = 200
for i in range(NLOOPS):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
