from yt.utilities.lib.misc_utilities import obtain_relative_velocity_vector

from min_yt import Dataset, FieldInfoContainer


def setup_fluid_fields(registry):
    def create_vector_fields(registry) -> None:
        def foo_closure(field, data):
            obtain_relative_velocity_vector(data, (xn, yn, zn), "bulk_velocity")

        xn, yn, zn = (("gas", f"velocity_{ax}") for ax in "xyz")

        registry.add_field(("gas", "velocity_spherical_radius"), function=foo_closure)

    create_vector_fields(registry)


def foo():
    ds = Dataset()

    ds.field_dependencies = {}
    ds.derived_field_list = []
    ds.field_info = FieldInfoContainer(ds)
    ds.coordinates.setup_fields(ds.field_info)
    setup_fluid_fields(ds.field_info)
    ds.field_info.check_derived_fields()


NLOOPS = 2000
for i in range(1, NLOOPS + 1):
    foo()
    print(f"{i}/{NLOOPS}", end="\r")
