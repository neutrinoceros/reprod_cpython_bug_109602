from mini_yt import Coordinates, Dataset
from mini_yt.lib import bar


def foo():
    ds = Dataset()
    coordinates = Coordinates()
    coordinates.setup_fields(ds.field_info)

    def foo_closure(field, data):
        bar(data, (xn, yn, zn))

    xn, yn, zn = (("gas", f"velocity_{ax}") for ax in "xyz")
    ds.field_info.add_field(("gas", "velocity_spherical_radius"), function=foo_closure)
    ds.field_info.check_derived_fields()


NLOOPS = 3000
for i in range(1, NLOOPS + 1):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
