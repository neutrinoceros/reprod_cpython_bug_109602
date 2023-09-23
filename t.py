import numpy as np

from min_yt import load_uniform_grid


def foo():
    shape = (16, 16, 16)
    ds = load_uniform_grid(
        data={("gas", "density"): np.ones(shape)},
        domain_dimensions=shape,
    )
    # inlined ds.all_data
    ds.index
    c = (ds.domain_right_edge + ds.domain_left_edge) / 2.0
    data_source = ds.region(c, ds.domain_left_edge, ds.domain_right_edge)
    data_source["gas", "density"].min()


NLOOPS = 200
for i in range(NLOOPS):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
