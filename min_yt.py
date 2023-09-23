from __future__ import annotations

import abc
import contextlib
import functools
import os
import weakref
from collections import UserDict, defaultdict
from collections.abc import Callable
from itertools import chain

import numpy as np
from unyt import Unit
from yt.frontends.stream.io import IOHandlerStream
from yt.geometry.coordinates.api import CartesianCoordinateHandler
from yt.geometry.geometry_handler import Index
from yt.units import dimensions
from yt.units.unit_registry import UnitRegistry  # type: ignore
from yt.units.unit_systems import unit_system_registry
from yt.units.yt_array import YTArray, YTQuantity
from yt.utilities.exceptions import YTFieldNotFound
from yt.utilities.lib.misc_utilities import obtain_relative_velocity_vector


def TranslationFunc(field_name):
    def _TranslationFunc(field, data):
        # We do a bunch of in-place modifications, so we will copy this.
        return data[field_name].copy()

    _TranslationFunc.alias_name = field_name
    return _TranslationFunc


class FieldDetector(defaultdict):
    def __init__(self, nd=16, ds=None, flat=False):
        self.nd = nd
        self.flat = flat
        self._spatial = not flat
        self.ActiveDimensions = [nd, nd, nd]
        self.shape = tuple(self.ActiveDimensions)
        self.size = np.prod(self.ActiveDimensions)
        self.LeftEdge = [0.0, 0.0, 0.0]
        self.RightEdge = [1.0, 1.0, 1.0]
        self.dds = np.ones(3, "float64")
        self.field_parameters = {}

        self.ds = ds

        class fake_index:
            class fake_io:
                pass

            io = fake_io()

        self.index = fake_index()
        self.requested = []
        self.requested_parameters = []
        defaultdict.__init__(
            self,
            lambda: np.ones((nd, nd, nd), dtype="float64")
            + 1e-4 * np.random.random((nd, nd, nd)),
        )

    def _reshape_vals(self, arr):
        return arr

    def __missing__(self, item: tuple[str, str] | str):
        field = item
        finfo = self.ds._get_field_info(field)
        params, permute_params = finfo._get_needed_parameters(self)
        self.field_parameters.update(params)
        # For those cases where we are guessing the field type, we will
        # need to re-update -- otherwise, our item will always not have the
        # field type.  This can lead to, for instance, "unknown" particle
        # types not getting correctly identified.
        # Note that the *only* way this works is if we also fix our field
        # dependencies during checking.  Bug #627 talks about this.
        _item: tuple[str, str] = finfo.name

        if not permute_params:
            vv = finfo(self)
        if vv is not None:
            self[_item] = vv
            return self[_item]

    @property
    def fcoords(self):
        fc = np.array(
            np.mgrid[0 : 1 : self.nd * 1j, 0 : 1 : self.nd * 1j, 0 : 1 : self.nd * 1j]
        )
        fc = fc.transpose()
        return self.ds.arr(fc, units="code_length")

    @property
    def fcoords_vertex(self):
        fc = np.random.random((self.nd, self.nd, self.nd, 8, 3))
        return self.ds.arr(fc, units="code_length")

    @property
    def fwidth(self):
        fw = np.ones((self.nd**3, 3), dtype="float64") / self.nd
        if not self.flat:
            fw.shape = (self.nd, self.nd, self.nd, 3)
        return self.ds.arr(fw, units="code_length")


class DerivedField:
    def __init__(
        self,
        name,
        sampling_type,
        function,
        units=None,
        display_field=True,
        not_in_all=False,
        display_name=None,
        ds=None,
        *,
        alias: DerivedField | None = None,
    ):
        self.name = name
        self.display_name = display_name
        self.not_in_all = not_in_all
        self.display_field = display_field
        self.sampling_type = sampling_type
        self.ds = ds

        self._function = function
        self.validators = []
        self.units = str(units)

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


class FieldInfoContainer(UserDict):
    known_other_fields = ()
    known_particle_fields = ()
    extra_union_fields = ()

    def __init__(self, ds, field_list, slice_info=None):
        super().__init__()
        self.ds = ds
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
        name,
        function: Callable,
        sampling_type: str,
        *,
        alias=None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("ds", self.ds)
        self[name] = DerivedField(name, sampling_type, function, alias=alias, **kwargs)

    def load_all_plugins(self, ftype: str | None = "gas") -> None:
        loaded = []
        for n in sorted(MINIMAL_FIELD_PLUGINS):
            f = MINIMAL_FIELD_PLUGINS[n]
            orig = set(self.items())
            f(self, ftype, slice_info=self.slice_info)
            loaded += [n for n, v in set(self.items()).difference(orig)]

        deps, unavailable = self.check_derived_fields(loaded)
        self.ds.field_dependencies.update(deps)
        # Note we may have duplicated
        dfl = set(self.ds.derived_field_list).union(deps.keys())
        self.ds.derived_field_list = sorted(dfl)
        return loaded, unavailable

    def alias(self, alias_name, original_name):
        # We default to CGS here, but in principle, this can be pluggable
        # as well.

        # self[original_name].units may be set to `None` at this point
        # to signal that units should be autoset later
        oru = self[original_name].units
        u = Unit(oru, registry=self.ds.unit_registry)
        units = str(self.ds.unit_system[u.dimensions])

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

    def check_derived_fields(self, fields_to_check=None):
        deps = {}
        unavailable = []
        fields_to_check = fields_to_check or list(self.keys())
        for field in fields_to_check:
            fi = self[field]
            try:
                fd = fi.get_dependencies(ds=self.ds)
            except (NotImplementedError, YTFieldNotFound):
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


class StreamHierarchy(GridIndex):
    grid = object

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

    def _setup_data_io(self):
        self.io = IOHandlerStream(self.ds)


class StreamFieldInfo(FieldInfoContainer):
    known_other_fields = (("density", ("code_mass/code_length**3", ["density"], None)),)

    def setup_fluid_fields(self):
        species_names = []
        for field in self.ds.stream_handler.field_units:
            self.ds.stream_handler.field_units[field]

        self.species_names = sorted(species_names)


class Dataset:
    default_fluid_type = "gas"
    fluid_types = ("gas", "deposit", "index")
    coordinates = None
    storage_filename = None
    _particle_type_counts = None
    _proj_type = "quad_proj"
    _determined_fields = None

    # the point in index space "domain_left_edge" doesn't necessarily
    # map to (0, 0, 0)
    domain_offset = np.zeros(3, dtype="int64")

    def __init__(self, *, stream_handler):
        self.fluid_types += ("stream",)
        self.stream_handler = stream_handler
        self.filename = self.stream_handler.name
        filename = "1234567890"
        dataset_type = "stream"
        # We return early and do NOT initialize a second time if this file has
        # already been initialized.
        self.dataset_type = dataset_type
        self.conversion_factors = {}
        self.parameters = {}
        self.field_units = {}
        self._determined_fields = {}

        self._input_filename: str = os.fspath(filename)

        self.unit_registry = UnitRegistry(unit_system="cgs")
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

        self.domain_left_edge = stream_handler.domain_left_edge.copy()
        self.domain_right_edge = stream_handler.domain_right_edge.copy()
        self.refine_by = stream_handler.refine_by
        self.dimensionality = stream_handler.dimensionality
        self._periodicity = (True, True, True)
        self.domain_dimensions = stream_handler.domain_dimensions
        self.current_time = stream_handler.simulation_time
        self.gamma = 5.0 / 3.0
        self.current_redshift = 0.0
        self.omega_lambda = 0.0
        self.omega_matter = 0.0
        self.hubble_constant = 0.0
        self.cosmological_simulation = 0

        self.unit_system = unit_system_registry["cgs"]
        self.unit_registry.unit_system = self.unit_system

        self.coordinates = CartesianCoordinateHandler(self)

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

    def _get_field_info(self, field, /):
        ftype, fname = field
        if (ftype, fname) in self.field_info:
            return self.field_info[ftype, fname]
        else:
            raise YTFieldNotFound(field, ds=self)

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
        self._quan = functools.partial(YTQuantity, registry=self.unit_registry)
        return self._quan


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

    return Dataset(stream_handler=handler)


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
