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
    ds.create_field_info()


NLOOPS = 200
for i in range(NLOOPS):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
