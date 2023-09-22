import time
import uuid
from collections import UserDict
from functools import cached_property
from itertools import chain

import numpy as np
from yt.data_objects.static_output import Dataset, _cached_datasets
from yt.frontends.stream.api import StreamFieldInfo, StreamHierarchy
from yt.geometry.api import Geometry


def setdefaultattr(obj, name, value):
    """Set attribute with *name* on *obj* with *value* if it doesn't exist yet

    Analogous to dict.setdefault
    """
    if not hasattr(obj, name):
        setattr(obj, name, value)
    return getattr(obj, name)


class StreamDictFieldHandler(UserDict):
    @property
    def all_fields(self):
        fields = chain.from_iterable(s.keys() for s in self.values())
        return list(set(fields))


class StreamHandler:
    def __init__(
        self,
        *,
        left_edges,
        right_edges,
        dimensions,
        levels,
        parent_ids,
        particle_count,
        processor_ids,
        fields,
        field_units,
        code_units,
        io=None,
        periodicity=(True, True, True),
        cell_widths=None,
    ):
        self.left_edges = np.array(left_edges)
        self.right_edges = np.array(right_edges)
        self.dimensions = dimensions
        self.levels = levels
        self.parent_ids = parent_ids
        self.particle_count = particle_count
        self.processor_ids = processor_ids
        self.num_grids = self.levels.size
        self.fields = fields
        self.field_units = field_units
        self.code_units = code_units
        self.io = io
        self.particle_types = {}
        self.periodicity = periodicity
        self.cell_widths = cell_widths
        self.parameters = {}

    def get_fields(self):
        return self.fields.all_fields


class MinimalStreamDataset(Dataset):
    _index_class = StreamHierarchy
    _field_info_class = StreamFieldInfo
    _dataset_type = "stream"

    def __init__(
        self,
        stream_handler,
        storage_filename=None,
        geometry="cartesian",
        unit_system="cgs",
        default_species_fields=None,
        *,
        axis_order=None,
    ):
        self.fluid_types += ("stream",)
        self.geometry = Geometry(geometry)
        self.stream_handler = stream_handler
        self._find_particle_types()
        name = f"InMemoryParameterFile_{uuid.uuid4().hex}"

        if geometry == "spectral_cube":
            # mimic FITSDataset specific interface to allow testing with
            # fake, in memory data
            setdefaultattr(self, "lon_axis", 0)
            setdefaultattr(self, "lat_axis", 1)
            setdefaultattr(self, "spec_axis", 2)
            setdefaultattr(self, "lon_name", "X")
            setdefaultattr(self, "lat_name", "Y")
            setdefaultattr(self, "spec_name", "z")
            setdefaultattr(self, "spec_unit", "")
            setdefaultattr(
                self,
                "pixel2spec",
                lambda pixel_value: self.arr(pixel_value, self.spec_unit),  # type: ignore [attr-defined]
            )
            setdefaultattr(
                self,
                "spec2pixel",
                lambda spec_value: self.arr(spec_value, "code_length"),
            )

        _cached_datasets[name] = self
        Dataset.__init__(
            self,
            name,
            self._dataset_type,
            unit_system=unit_system,
            default_species_fields=default_species_fields,
            axis_order=axis_order,
        )

    @property
    def filename(self):
        return self.stream_handler.name

    @cached_property
    def unique_identifier(self) -> str:
        return str(self.parameters["CurrentTimeIdentifier"])

    def _parse_parameter_file(self):
        self.parameters["CurrentTimeIdentifier"] = time.time()
        self.domain_left_edge = self.stream_handler.domain_left_edge.copy()
        self.domain_right_edge = self.stream_handler.domain_right_edge.copy()
        self.refine_by = self.stream_handler.refine_by
        self.dimensionality = self.stream_handler.dimensionality
        self._periodicity = self.stream_handler.periodicity
        self.domain_dimensions = self.stream_handler.domain_dimensions
        self.current_time = self.stream_handler.simulation_time
        self.gamma = 5.0 / 3.0
        self.parameters["EOSType"] = -1
        self.parameters["CosmologyHubbleConstantNow"] = 1.0
        self.parameters["CosmologyCurrentRedshift"] = 1.0
        self.parameters["HydroMethod"] = -1
        self.parameters.update(self.stream_handler.parameters)

        self.current_redshift = 0.0
        self.omega_lambda = 0.0
        self.omega_matter = 0.0
        self.hubble_constant = 0.0
        self.cosmological_simulation = 0

    def _set_units(self):
        self.field_units = self.stream_handler.field_units

    def _set_code_unit_attributes(self):
        base_units = self.stream_handler.code_units
        attrs = (
            "length_unit",
            "mass_unit",
            "time_unit",
            "velocity_unit",
            "magnetic_unit",
        )
        for unit, attr in zip(base_units, attrs):
            if unit == "code_magnetic":
                # If no magnetic unit was explicitly specified
                # we skip it now and take care of it at the bottom
                continue
            else:
                uq = self.quan(1.0, unit)
            setattr(self, attr, uq)
        if not hasattr(self, "magnetic_unit"):
            self.magnetic_unit = np.sqrt(
                4 * np.pi * self.mass_unit / (self.time_unit**2 * self.length_unit)
            )

    @classmethod
    def _is_valid(cls, filename: str, *args, **kwargs) -> bool:
        return False

    @property
    def _skip_cache(self):
        return True

    def _find_particle_types(self):
        particle_types = set()
        for k, v in self.stream_handler.particle_types.items():
            if v:
                particle_types.add(k[0])
        self.particle_types = tuple(particle_types)
        self.particle_types_raw = self.particle_types


def load_uniform_grid(
    *,
    data,
    domain_dimensions,
):
    domain_dimensions = np.array(domain_dimensions)
    bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], "float64")
    domain_left_edge = np.array(bbox[:, 0], "float64")
    domain_right_edge = np.array(bbox[:, 1], "float64")
    grid_levels = np.zeros(1, dtype="int32").reshape((1, 1))
    # First we fix our field names, apply units to data
    # and check for consistency of field shapes

    new_data = {}
    for field, val in data.items():
        new_data[field] = val.copy()
    data = new_data

    field_units = {field: "" for field in data}

    sfh = StreamDictFieldHandler()
    sfh.update({0: data})
    grid_left_edges = domain_left_edge
    grid_right_edges = domain_right_edge
    grid_dimensions = domain_dimensions.reshape(1, 3).astype("int32")

    handler = StreamHandler(
        left_edges=grid_left_edges,
        right_edges=grid_right_edges,
        dimensions=grid_dimensions,
        levels=grid_levels,
        parent_ids=-np.ones(1, dtype="int64"),
        particle_count=np.zeros(1, dtype="int64").reshape(1, 1),
        processor_ids=np.zeros(1).reshape((1, 1)),
        fields=sfh,
        field_units=field_units,
        code_units=(
            "code_length",
            "code_mass",
            "code_time",
            "code_velocity",
            "code_magnetic",
        ),
    )

    handler.name = "UniformGridData"
    handler.domain_left_edge = domain_left_edge
    handler.domain_right_edge = domain_right_edge
    handler.refine_by = 2

    handler.dimensionality = 3
    handler.domain_dimensions = domain_dimensions
    handler.simulation_time = 1.0
    handler.cosmology_simulation = 0

    return MinimalStreamDataset(
        handler,
        geometry="cartesian",
        axis_order=None,
        unit_system="cgs",
        default_species_fields=None,
    )
