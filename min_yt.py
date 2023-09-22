from collections import UserDict
from itertools import chain

import numpy as np

from yt.frontends.stream.data_structures import StreamDataset

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
        code_units=("code_length", "code_mass", "code_time", "code_velocity", "code_magnetic"),
    )

    handler.name = "UniformGridData"
    handler.domain_left_edge = domain_left_edge 
    handler.domain_right_edge = domain_right_edge
    handler.refine_by = 2

    handler.dimensionality = 3 
    handler.domain_dimensions = domain_dimensions
    handler.simulation_time = 1.0
    handler.cosmology_simulation = 0 

    return StreamDataset(
        handler,
        geometry="cartesian",
        axis_order=None,
        unit_system="cgs",
        default_species_fields=None,
    )
