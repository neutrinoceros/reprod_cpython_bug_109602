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
    # Now we do things that we need an instantiated index for
    # ...first off, we create our field_info now.
    oldsettings = np.geterr()
    np.seterr(all="ignore")
    ds.create_field_info()
    np.seterr(**oldsettings)


NLOOPS = 200
for i in range(NLOOPS):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
