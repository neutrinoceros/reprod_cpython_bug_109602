import numpy as np
from yt import load_uniform_grid
from yt.fields.field_plugin_registry import register_field_plugin
from yt.utilities.lib.misc_utilities import obtain_relative_velocity_vector


def create_vector_fields(
    registry,
    basename,
    field_units,
    ftype="gas",
) -> None:
    axis_order = registry.ds.coordinates.axis_order

    # LEAK
    xn, yn, zn = ((ftype, f"{basename}_{ax}") for ax in axis_order)

    def foo_closure(field, data):
        obtain_relative_velocity_vector(data, (xn, yn, zn), f"bulk_{basename}")

    registry.add_field(
        (ftype, f"{basename}_spherical_radius"),
        sampling_type="local",
        function=foo_closure,
        units=field_units,
    )


@register_field_plugin
def setup_fluid_fields(registry, ftype="gas", slice_info=None):
    unit_system = registry.ds.unit_system

    create_vector_fields(registry, "velocity", unit_system["velocity"], ftype)


def main():
    shape = (16, 16, 16)
    ds = load_uniform_grid(
        data={("gas", "density"): np.ones(shape)},
        domain_dimensions=shape,
    )

    data_source = ds.all_data()
    data_source["gas", "density"].min()


NLOOPS = 200
for i in range(NLOOPS):
    main()
    print(f"{i}/{NLOOPS}", end="\r")
