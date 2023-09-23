import numpy as np

from min_yt import load_uniform_grid


def foo():
    shape = (16, 16, 16)
    ds = load_uniform_grid(
        data={("gas", "density"): np.ones(shape)},
        domain_dimensions=shape,
    )

    data_source = ds.all_data()
    data_source["gas", "density"].min()


NLOOPS = 100
for i in range(NLOOPS):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
