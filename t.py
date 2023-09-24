from min_yt import Dataset, FieldInfoContainer


def foo():
    ds = Dataset()

    ds.field_dependencies = {}
    ds.derived_field_list = []
    ds.field_info = FieldInfoContainer(ds)
    ds.coordinates.setup_fields(ds.field_info)
    ds.field_info.setup_fluid_index_fields()

    ds.field_info.load_all_plugins()
    deps = ds.field_info.check_derived_fields()


NLOOPS = 1500
for i in range(1, NLOOPS + 1):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
