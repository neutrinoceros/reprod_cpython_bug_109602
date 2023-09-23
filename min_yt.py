from __future__ import annotations

import abc
import contextlib
import functools
import os
import weakref
from collections import UserDict
from collections.abc import Callable
from itertools import chain
from typing import Any, Literal, Optional, Union

import numpy as np
from unyt import Unit, UnitSystem
from unyt.exceptions import UnitConversionError
from yt.arraytypes import blankRecordArray
from yt.data_objects.index_subobjects.grid_patch import AMRGridPatch
from yt.data_objects.region_expression import RegionExpression
from yt.fields.derived_field import NullFunc, TranslationFunc
from yt.fields.field_detector import FieldDetector
from yt.fields.field_exceptions import NeedsConfiguration
from yt.fields.field_plugin_registry import FunctionName
from yt.geometry.coordinates.api import CartesianCoordinateHandler
from yt.geometry.geometry_handler import Index
from yt.units import YTQuantity, dimensions
from yt.units.dimensions import current_mks, dimensionless
from yt.units.unit_registry import UnitRegistry  # type: ignore
from yt.units.unit_systems import create_code_unit_system, unit_system_registry
from yt.units.yt_array import YTArray
from yt.utilities.definitions import MAXLEVEL
from yt.utilities.exceptions import (
    YTCoordinateNotImplemented,
    YTDomainOverflow,
    YTFieldNotFound,
)
from yt.utilities.io_handler import io_registry
from yt.utilities.lib.misc_utilities import obtain_relative_velocity_vector
from yt.utilities.object_registries import data_object_registry


class DerivedField:
    def __init__(
        self,
        name: FieldKey,
        sampling_type,
        function,
        units: Optional[Union[str, bytes, Unit]] = None,
        vector_field=False,
        display_field=True,
        not_in_all=False,
        display_name=None,
        dimensions=None,
        ds=None,
        *,
        alias: Optional["DerivedField"] = None,
    ):
        self.name = name
        self.display_name = display_name
        self.not_in_all = not_in_all
        self.display_field = display_field
        self.sampling_type = sampling_type
        self.vector_field = vector_field
        self.ds = ds

        self._function = function
        self.validators = []

        # handle units
        if isinstance(units, str):
            self.units = units
        else:
            self.units = str(units)

        self.output_units = self.units
        self.dimensions = dimensions

        if alias is None:
            self._shared_aliases_list = [self]
        else:
            self._shared_aliases_list = alias._shared_aliases_list
            self._shared_aliases_list.append(self)

    def get_dependencies(self, *args, **kwargs):
        """
        This returns a list of names of fields that this field depends on.
        """
        e = FieldDetector(*args, **kwargs)
        e[self.name]
        return e

    def _get_needed_parameters(self, fd):
        return {}, {}

    _unit_registry = None

    @contextlib.contextmanager
    def unit_registry(self, data):
        old_registry = self._unit_registry
        ur = data.ds.unit_registry
        self._unit_registry = ur
        yield
        self._unit_registry = old_registry

    def __call__(self, data):
        """Return the value of the field in a given *data* object."""
        with self.unit_registry(data):
            dd = self._function(self, data)
        return dd


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
        self.io = None
        self.cell_widths = None
        self.parameters = {}

    def get_fields(self):
        return self.fields.all_fields


class Dataset(abc.ABC):
    default_fluid_type = "gas"
    fluid_types: tuple[FieldType, ...] = ("gas", "deposit", "index")
    coordinates = None
    storage_filename = None
    _index_class: type[Index]
    _particle_type_counts = None
    _proj_type = "quad_proj"
    _determined_fields: Optional[dict[str, list[FieldKey]]] = None

    # the point in index space "domain_left_edge" doesn't necessarily
    # map to (0, 0, 0)
    domain_offset = np.zeros(3, dtype="int64")
    _force_periodicity = False

    def __init__(
        self,
        filename: str,
        dataset_type: Optional[str] = None,
        units_override: Optional[dict[str, str]] = None,
        # valid unit_system values include all keys from unyt.unit_systems.unit_systems_registry + "code"
        unit_system="cgs",
        default_species_fields: Optional[
            "Any"
        ] = None,  # Any used as a placeholder here
        *,
        axis_order: Optional[AxisOrder] = None,
    ) -> None:
        # We return early and do NOT initialize a second time if this file has
        # already been initialized.
        self.dataset_type = dataset_type
        self.conversion_factors = {}
        self.parameters: dict[str, Any] = {}
        self.region_expression = self.r = RegionExpression(self)
        self.field_units = {}
        self._determined_fields = {}
        self.units_override = units_override
        self.default_species_fields = default_species_fields

        self._input_filename: str = os.fspath(filename)

        self.no_cgs_equiv_length = False
        self._create_unit_registry(unit_system)

        self._parse_parameter_file()
        self.set_units()
        self._assign_unit_system(unit_system)
        self._setup_coordinate_handler(axis_order)
        self._set_derived_attrs()
        self._setup_classes()

    def _set_derived_attrs(self):
        self.domain_center = 0.5 * (self.domain_right_edge + self.domain_left_edge)
        self.domain_width = self.domain_right_edge - self.domain_left_edge
        self.current_time = self.quan(self.current_time, "code_time")
        for attr in ("center", "width", "left_edge", "right_edge"):
            n = f"domain_{attr}"
            v = getattr(self, n)
            # Note that we don't add on _ipython_display_ here because
            # everything is stored inside a MutableAttribute.
            v = self.arr(v, "code_length")
            setattr(self, n, v)

    _instantiated_index = None

    @property
    def index(self):
        return self._instantiated_index

    @property
    def field_list(self):
        return self.index.field_list

    def _setup_coordinate_handler(self, axis_order: Optional[AxisOrder]) -> None:
        self.coordinates = CartesianCoordinateHandler(self, ordering=axis_order)

    def _get_field_info(self, field, /):
        field_info, candidates = self._get_field_info_helper(field)
        return field_info

    def _get_field_info_helper(self, field, /):
        self.index
        ftype, fname = field

        if (ftype, fname) in self.field_info:
            return self.field_info[ftype, fname], []

        raise YTFieldNotFound(field, ds=self)

    def _setup_classes(self):
        # Called by subclass
        self.object_types = []
        self.objects = []
        self.plots = []
        for name, cls in sorted(data_object_registry.items()):
            self._add_object_class(name, cls)
        self.object_types.sort()

    def _add_object_class(self, name, base):
        # skip projection data objects that don't make sense
        # for this type of data
        if "proj" in name and name != self._proj_type:
            return
        elif "proj" in name:
            name = "proj"
        self.object_types.append(name)
        obj = functools.partial(base, ds=weakref.proxy(self))
        obj.__doc__ = base.__doc__
        setattr(self, name, obj)

    def _assign_unit_system(
        self,
        # valid unit_system values include all keys from unyt.unit_systems.unit_systems_registry + "code"
        unit_system: Literal[
            "cgs",
            "mks",
            "imperial",
            "galactic",
            "solar",
            "geometrized",
            "planck",
            "code",
        ],
    ) -> None:
        # we need to determine if the requested unit system
        # is mks-like: i.e., it has a current with the same
        # dimensions as amperes.
        mks_system = False
        us = unit_system_registry[str(unit_system).lower()]
        mks_system = us.base_units[current_mks] is not None

        current_mks_unit = "A" if mks_system else None
        us = create_code_unit_system(
            self.unit_registry, current_mks_unit=current_mks_unit
        )
        us = unit_system_registry[str(unit_system).lower()]

        self._unit_system_name: str = unit_system

        self.unit_system: UnitSystem = us
        self.unit_registry.unit_system = self.unit_system

    def _create_unit_registry(self, unit_system):
        from yt.units import dimensions

        # yt assumes a CGS unit system by default (for back compat reasons).
        # Since unyt is MKS by default we specify the MKS values of the base
        # units in the CGS system. So, for length, 1 cm = .01 m. And so on.
        # Note that the values associated with the code units here will be
        # modified once we actually determine what the code units are from
        # the dataset
        # NOTE that magnetic fields are not done here yet, see set_code_units
        self.unit_registry = UnitRegistry(unit_system=unit_system)
        # 1 cm = 0.01 m
        self.unit_registry.add("code_length", 0.01, dimensions.length)
        # 1 g = 0.001 kg
        self.unit_registry.add("code_mass", 0.001, dimensions.mass)
        # 1 g/cm**3 = 1000 kg/m**3
        self.unit_registry.add("code_density", 1000.0, dimensions.density)
        # 1 erg/g = 1.0e-4 J/kg
        self.unit_registry.add(
            "code_specific_energy", 1.0e-4, dimensions.energy / dimensions.mass
        )
        # 1 s = 1 s
        self.unit_registry.add("code_time", 1.0, dimensions.time)
        # 1 K = 1 K
        self.unit_registry.add("code_temperature", 1.0, dimensions.temperature)
        # 1 dyn/cm**2 = 0.1 N/m**2
        self.unit_registry.add("code_pressure", 0.1, dimensions.pressure)
        # 1 cm/s = 0.01 m/s
        self.unit_registry.add("code_velocity", 0.01, dimensions.velocity)
        # metallicity
        self.unit_registry.add("code_metallicity", 1.0, dimensions.dimensionless)
        # dimensionless hubble parameter
        self.unit_registry.add("h", 1.0, dimensions.dimensionless, r"h")
        # cosmological scale factor
        self.unit_registry.add("a", 1.0, dimensions.dimensionless)

    def set_units(self):
        """
        Creates the unit registry for this dataset.

        """
        self.set_code_units()

    def set_code_units(self):
        # set attributes like ds.length_unit
        self._set_code_unit_attributes()

        self.unit_registry.modify("code_length", self.length_unit)
        self.unit_registry.modify("code_mass", self.mass_unit)
        self.unit_registry.modify("code_time", self.time_unit)
        vel_unit = getattr(self, "velocity_unit", self.length_unit / self.time_unit)
        pressure_unit = getattr(
            self,
            "pressure_unit",
            self.mass_unit / (self.length_unit * (self.time_unit) ** 2),
        )
        temperature_unit = getattr(self, "temperature_unit", 1.0)
        density_unit = getattr(
            self, "density_unit", self.mass_unit / self.length_unit**3
        )
        specific_energy_unit = getattr(self, "specific_energy_unit", vel_unit**2)
        self.unit_registry.modify("code_velocity", vel_unit)
        self.unit_registry.modify("code_temperature", temperature_unit)
        self.unit_registry.modify("code_pressure", pressure_unit)
        self.unit_registry.modify("code_density", density_unit)
        self.unit_registry.modify("code_specific_energy", specific_energy_unit)
        # Defining code units for magnetic fields are tricky because
        # they have different dimensions in different unit systems, so we have
        # to handle them carefully
        # We have to cast this explicitly to MKS base units, otherwise
        # unyt will convert it automatically to Tesla
        value = self.magnetic_unit.to_value("sqrt(kg)/(sqrt(m)*s)")
        dims = dimensions.magnetic_field_cgs

        self.unit_registry.add("code_magnetic", value, dims)
        # domain_width does not yet exist
        if self.domain_left_edge is not None and self.domain_right_edge is not None:
            DW = self.arr(self.domain_right_edge - self.domain_left_edge, "code_length")
            self.unit_registry.add(
                "unitary", float(DW.max() * DW.units.base_value), DW.units.dimensions
            )

    _arr = None

    @property
    def arr(self):
        if self._arr is not None:
            return self._arr
        self._arr = functools.partial(YTArray, registry=self.unit_registry)
        return self._arr

    _quan = None

    @property
    def quan(self):
        if self._quan is not None:
            return self._quan
        self._quan = functools.partial(YTQuantity, registry=self.unit_registry)
        return self._quan


class FieldInfoContainer(UserDict):
    fallback = None
    known_other_fields: KnownFieldsT = ()
    known_particle_fields: KnownFieldsT = ()
    extra_union_fields: tuple[FieldKey, ...] = ()

    def __init__(self, ds, field_list: list[FieldKey], slice_info=None):
        super().__init__()
        self._show_field_errors: list[Exception] = []
        self.ds = ds
        # Now we start setting things up.
        self.field_list = field_list
        self.slice_info = slice_info
        self.field_aliases = {}

    def setup_fluid_index_fields(self):
        # Now we get all our index types and set up aliases to them
        index_fields = {f for _, f in self if _ == "index"}
        for ftype in self.ds.fluid_types:
            if ftype in ("index", "deposit"):
                continue
            for f in index_fields:
                self.alias((ftype, f), ("index", f))

    def add_field(
        self,
        name: FieldKey,
        function: Callable,
        sampling_type: str,
        *,
        alias=None,
        force_override: bool = False,
        **kwargs,
    ) -> None:
        # Handle the case where the field has already been added.
        if not force_override and name in self:
            return

        kwargs.setdefault("ds", self.ds)
        self[name] = DerivedField(name, sampling_type, function, alias=alias, **kwargs)

    def load_all_plugins(self, ftype: Optional[str] = "gas") -> None:
        loaded = []
        for n in sorted(MINIMAL_FIELD_PLUGINS):
            loaded += self.load_plugin(n, ftype)
        self.find_dependencies(loaded)

    def load_plugin(
        self,
        plugin_name: FunctionName,
        ftype: FieldType = "gas",
        skip_check: bool = False,
    ):
        f = MINIMAL_FIELD_PLUGINS[plugin_name]
        orig = set(self.items())
        f(self, ftype, slice_info=self.slice_info)
        loaded = [n for n, v in set(self.items()).difference(orig)]
        return loaded

    def find_dependencies(self, loaded):
        deps, unavailable = self.check_derived_fields(loaded)
        self.ds.field_dependencies.update(deps)
        # Note we may have duplicated
        dfl = set(self.ds.derived_field_list).union(deps.keys())
        self.ds.derived_field_list = sorted(dfl)
        return loaded, unavailable

    def add_output_field(self, name, sampling_type, **kwargs):
        kwargs.setdefault("ds", self.ds)
        self[name] = DerivedField(name, sampling_type, NullFunc, **kwargs)

    def alias(
        self,
        alias_name: FieldKey,
        original_name: FieldKey,
        units: Optional[str] = None,
    ):
        if units is None:
            # We default to CGS here, but in principle, this can be pluggable
            # as well.

            # self[original_name].units may be set to `None` at this point
            # to signal that units should be autoset later
            oru = self[original_name].units

            u = Unit(oru, registry=self.ds.unit_registry)
            if u.dimensions is not dimensionless:
                units = str(self.ds.unit_system[u.dimensions])
            else:
                units = oru

        self.field_aliases[alias_name] = original_name
        function = TranslationFunc(original_name)

        self.add_field(
            alias_name,
            function=function,
            sampling_type=self[original_name].sampling_type,
            display_name=self[original_name].display_name,
            units=units,
            alias=self[original_name],
        )

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if self.fallback is None:
            return False

    def check_derived_fields(self, fields_to_check=None):
        # The following exceptions lists were obtained by expanding an
        # all-catching `except Exception`.
        # We define
        # - a blacklist (exceptions that we know should be caught)
        # - a whitelist (exceptions that should be handled)
        # - a greylist (exceptions that may be covering bugs but should be checked)
        # See https://github.com/yt-project/yt/issues/2853
        # in the long run, the greylist should be removed
        whitelist = (NotImplementedError,)
        greylist = (
            YTFieldNotFound,
            YTDomainOverflow,
            YTCoordinateNotImplemented,
            NeedsConfiguration,
            TypeError,
            ValueError,
            IndexError,
            AttributeError,
            KeyError,
            # code smells -> those are very likely bugs
            UnitConversionError,  # solved in GH PR 2897 ?
            # RecursionError is clearly a bug, and was already solved once
            # in GH PR 2851
            RecursionError,
        )

        deps = {}
        unavailable = []
        fields_to_check = fields_to_check or list(self.keys())
        for field in fields_to_check:
            fi = self[field]
            try:
                fd = fi.get_dependencies(ds=self.ds)
            except (*whitelist, *greylist):
                self.pop(field)
                continue
            # This next bit checks that we can't somehow generate everything.
            # We also manually update the 'requested' attribute
            fd.requested = set(fd.requested)
            deps[field] = fd

        # now populate the derived field list with results
        # this violates isolation principles and should be refactored
        dfl = set(self.ds.derived_field_list).union(deps.keys())
        dfl = sorted(dfl)

        if not hasattr(self.ds.index, "meshes"):
            # the meshes attribute characterizes an unstructured-mesh data structure

            # ideally this filtering should not be required
            # and this could maybe be handled in fi.get_dependencies
            # but it's a lot easier to do here

            filtered_dfl = []
            for field in dfl:
                ftype, fname = field
                if "vertex" in fname:
                    continue

                filtered_dfl.append(field)
            dfl = filtered_dfl

        self.ds.derived_field_list = dfl
        return deps, unavailable


class GridIndex(Index, abc.ABC):
    """The index class for patch and block AMR datasets."""

    float_type = "float64"
    _preload_implemented = False
    _index_properties = (
        "grid_left_edge",
        "grid_right_edge",
        "grid_levels",
        "grid_particle_count",
        "grid_dimensions",
    )

    def _setup_geometry(self):
        self._count_grids()
        self._initialize_grid_arrays()
        self._parse_index()
        self._populate_grid_objects()
        self._initialize_level_stats()

    def _initialize_grid_arrays(self):
        self.grid_dimensions = np.ones((self.num_grids, 3), "int32")
        self.grid_left_edge = self.ds.arr(
            np.zeros((self.num_grids, 3), self.float_type), "code_length"
        )
        self.grid_right_edge = self.ds.arr(
            np.ones((self.num_grids, 3), self.float_type), "code_length"
        )
        self.grid_levels = np.zeros((self.num_grids, 1), "int32")
        self.grid_particle_count = np.zeros((self.num_grids, 1), "int32")

    def _initialize_level_stats(self):
        # Now some statistics:
        #   0 = number of grids
        #   1 = number of cells
        #   2 = blank
        desc = {"names": ["numgrids", "numcells", "level"], "formats": ["int64"] * 3}
        self.level_stats = blankRecordArray(desc, MAXLEVEL)
        self.level_stats["level"] = list(range(MAXLEVEL))
        self.level_stats["numgrids"] = [0 for i in range(MAXLEVEL)]
        self.level_stats["numcells"] = [0 for i in range(MAXLEVEL)]
        for level in range(self.max_level + 1):
            self.level_stats[level]["numgrids"] = np.sum(self.grid_levels == level)
            li = self.grid_levels[:, 0] == level
            self.level_stats[level]["numcells"] = (
                self.grid_dimensions[li, :].prod(axis=1).sum()
            )

    _grid_chunksize = 1000


class StreamGrid(AMRGridPatch):
    """
    Class representing a single In-memory Grid instance.
    """

    __slots__ = ["proc_num"]
    _id_offset = 0

    def __init__(self, id, index):
        """
        Returns an instance of StreamGrid with *id*, associated with *filename*
        and *index*.
        """
        # All of the field parameters will be passed to us as needed.
        AMRGridPatch.__init__(self, id, filename=None, index=index)
        self._children_ids = []
        self._parent_id = -1
        self.Level = -1

    @property
    def Parent(self):
        return None


class StreamHierarchy(GridIndex):
    grid = StreamGrid

    def __init__(self, ds, dataset_type=None):
        self.dataset_type = dataset_type
        self.float_type = "float64"
        self.dataset = weakref.proxy(ds)  # for _obtain_enzo
        self.stream_handler = ds.stream_handler
        self.float_type = "float64"
        self.directory = os.getcwd()
        GridIndex.__init__(self, ds, dataset_type)

    def _count_grids(self):
        self.num_grids = self.stream_handler.num_grids

    def _parse_index(self):
        self.grid_dimensions = self.stream_handler.dimensions
        self.grid_left_edge[:] = self.stream_handler.left_edges
        self.grid_right_edge[:] = self.stream_handler.right_edges
        self.grid_levels[:] = self.stream_handler.levels
        self.min_level = self.grid_levels.min()
        self.grid_procs = self.stream_handler.processor_ids
        self.grid_particle_count[:] = self.stream_handler.particle_count
        self.grid_cell_widths = None
        self.grids = []
        # We enumerate, so it's 0-indexed id and 1-indexed pid
        for id in range(self.num_grids):
            self.grids.append(self.grid(id, self))
            self.grids[id].Level = self.grid_levels[id, 0]
        self.stream_handler.parent_ids.tolist()

        self.max_level = self.grid_levels.max()
        temp_grids = np.empty(self.num_grids, dtype="object")
        for i, grid in enumerate(self.grids):
            grid.filename = None
            grid._prepare_grid()
            grid._setup_dx()
            grid.proc_num = self.grid_procs[i]
            temp_grids[i] = grid
        self.grids = temp_grids

    def _initialize_grid_arrays(self):
        GridIndex._initialize_grid_arrays(self)
        self.grid_procs = np.zeros((self.num_grids, 1), "int32")

    def _detect_output_fields(self):
        # NOTE: Because particle unions add to the actual field list, without
        # having the keys in the field list itself, we need to double check
        # here.
        fl = set(self.stream_handler.get_fields())
        fl.update(set(getattr(self, "field_list", [])))
        self.field_list = list(fl)

    def _populate_grid_objects(self):
        for g in self.grids:
            g._setup_dx()
        self.max_level = self.grid_levels.max()

    def _setup_data_io(self):
        self.io = io_registry[self.dataset_type](self.ds)


class StreamFieldInfo(FieldInfoContainer):
    known_other_fields: KnownFieldsT = (
        ("density", ("code_mass/code_length**3", ["density"], None)),
    )

    def setup_fluid_fields(self):
        species_names = []
        for field in self.ds.stream_handler.field_units:
            self.ds.stream_handler.field_units[field]

        self.species_names = sorted(species_names)

    def add_output_field(self, name, sampling_type, **kwargs):
        if name in self.ds.stream_handler.field_units:
            kwargs["units"] = self.ds.stream_handler.field_units[name]
        super().add_output_field(name, sampling_type, **kwargs)


class MinimalStreamDataset(Dataset):
    _dataset_type = "stream"

    def __init__(self, *, stream_handler):
        self.fluid_types += ("stream",)
        self.stream_handler = stream_handler
        self.filename = self.stream_handler.name
        Dataset.__init__(
            self,
            filename="InMemoryParameterFile_1234567890",
            dataset_type=self._dataset_type,
            unit_system="cgs",
        )

    def _parse_parameter_file(self):
        self.domain_left_edge = self.stream_handler.domain_left_edge.copy()
        self.domain_right_edge = self.stream_handler.domain_right_edge.copy()
        self.refine_by = self.stream_handler.refine_by
        self.dimensionality = self.stream_handler.dimensionality
        self._periodicity = (True, True, True)
        self.domain_dimensions = self.stream_handler.domain_dimensions
        self.current_time = self.stream_handler.simulation_time
        self.gamma = 5.0 / 3.0
        self.current_redshift = 0.0
        self.omega_lambda = 0.0
        self.omega_matter = 0.0
        self.hubble_constant = 0.0
        self.cosmological_simulation = 0

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

    return MinimalStreamDataset(stream_handler=handler)


def setup_fluid_fields(registry, ftype="gas", slice_info=None):
    unit_system = registry.ds.unit_system

    def create_vector_fields(
        registry,
        basename,
        field_units,
        ftype="gas",
    ) -> None:
        axis_order = registry.ds.coordinates.axis_order

        xn, yn, zn = ((ftype, f"{basename}_{ax}") for ax in axis_order)

        def foo_closure(field, data):
            obtain_relative_velocity_vector(data, (xn, yn, zn), f"bulk_{basename}")

        registry.add_field(
            (ftype, f"{basename}_spherical_radius"),
            sampling_type="local",
            function=foo_closure,
            units=field_units,
        )

    create_vector_fields(registry, "velocity", unit_system["velocity"], ftype)


MINIMAL_FIELD_PLUGINS = {"fluid": setup_fluid_fields}
