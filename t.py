from mini_yt import Coordinates, Dataset
from mini_yt.lib import bar


def main():
    ds = Dataset()
    coordinates = Coordinates()
    coordinates.setup_fields(ds.field_info)
    ds.field_info.add_field("test", function=lambda data: bar(data))
    ds.field_info.check_derived_fields()


NLOOPS = 3000
for i in range(1, NLOOPS + 1):
    main()
    print(f"{i}/{NLOOPS}", end="\r")
