from mini_yt import Coordinates, Dataset
from mini_yt.lib import bar


def foo():
    ds = Dataset()
    coordinates = Coordinates()
    coordinates.setup_fields(ds.field_info)

    def _f(field, data):
        bar(data)

    ds.field_info.add_field(("a", "b"), function=_f)
    ds.field_info.check_derived_fields()


NLOOPS = 3000
for i in range(1, NLOOPS + 1):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
