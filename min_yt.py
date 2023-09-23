from __future__ import annotations

import abc
import functools
import hashlib
import itertools
import os
import time
import uuid
import weakref
from collections import UserDict, defaultdict
from collections.abc import Callable
from itertools import chain
from typing import Any, Literal, Optional, Union

import numpy as np
import yt.geometry.selection_routines
import yt.utilities.logger
from unyt import Unit, UnitSystem, unyt_quantity
from unyt.exceptions import UnitConversionError
from yt._typing import AnyFieldKey, FieldKey, FieldName, FieldType, KnownFieldsT
from yt.data_objects.derived_quantities import DerivedQuantityCollection
from yt.data_objects.field_data import YTFieldData
from yt.data_objects.region_expression import RegionExpression
from yt.fields.derived_field import (
    DerivedField,
    NullFunc,
    TranslationFunc,
)
from yt.fields.field_exceptions import NeedsConfiguration
from yt.fields.field_plugin_registry import FunctionName, field_plugins
from yt.fields.field_type_container import FieldTypeContainer
from yt.frontends.stream.api import StreamHierarchy
from yt.funcs import (
    iter_fields,
    obj_length,
)
from yt.geometry.api import Geometry
from yt.geometry.coordinates.api import CartesianCoordinateHandler
from yt.geometry.geometry_handler import Index
from yt.units import UnitContainer, dimensions
from yt.units.dimensions import (
    current_mks,
    dimensionless,  # type: ignore
)
from yt.units.unit_registry import UnitRegistry  # type: ignore
from yt.units.unit_systems import (
    create_code_unit_system,
    unit_system_registry,
)
from yt.units.yt_array import YTArray, YTQuantity
from yt.utilities.exceptions import (
    GenerationInProgress,
    YTCoordinateNotImplemented,
    YTDomainOverflow,
    YTFieldNotFound,
)
from yt.utilities.object_registries import data_object_registry


class YTDataContainer(abc.ABC):
    """
    Generic YTDataContainer container.  By itself, will attempt to
    generate field, read fields (method defined by derived classes)
    and deal with passing back and forth field parameters.
    """

    _chunk_info = None
    _num_ghost_zones = 0
    _con_args: tuple[str, ...] = ()
    _skip_add = False
    _container_fields: tuple[AnyFieldKey, ...] = ()
    _tds_attrs: tuple[str, ...] = ()
    _tds_fields: tuple[str, ...] = ()
    _field_cache = None
    _index = None
    _key_fields: list[str]

    def __init__(self, ds: Optional["Dataset"], field_parameters) -> None:
        """
        Typically this is never called directly, but only due to inheritance.
        It associates a :class:`~yt.data_objects.static_output.Dataset` with the class,
        sets its initial set of fields, and the remainder of the arguments
        are passed as field_parameters.
        """
        # ds is typically set in the new object type created in
        # Dataset._add_object_class but it can also be passed as a parameter to the
        # constructor, in which case it will override the default.
        # This code ensures it is never not set.
        self.ds = ds
        self._current_fluid_type = self.ds.default_fluid_type
        self.ds.objects.append(weakref.proxy(self))
        self.field_data = YTFieldData()

        mag_unit = "G"
        self._default_field_parameters = {
            "center": self.ds.arr(np.zeros(3, dtype="float64"), "cm"),
            "bulk_velocity": self.ds.arr(np.zeros(3, dtype="float64"), "cm/s"),
            "bulk_magnetic_field": self.ds.arr(np.zeros(3, dtype="float64"), mag_unit),
            "normal": self.ds.arr([0.0, 0.0, 1.0], ""),
        }
        if field_parameters is None:
            field_parameters = {}
        self._set_default_field_parameters()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if hasattr(cls, "_type_name") and not cls._skip_add:
            name = getattr(cls, "_override_selector_name", cls._type_name)
            data_object_registry[name] = cls

    @property
    def index(self):
        if self._index is not None:
            return self._index
        self._index = self.ds.index
        return self._index

    def _set_default_field_parameters(self):
        self.field_parameters = {}
        for k, v in self._default_field_parameters.items():
            self.field_parameters[k] = v

    def _set_center(self, center):
        self.center = center
        self.field_parameters["center"] = self.center

    def __getitem__(self, key):
        """
        Returns a single field.  Will add if necessary.
        """
        f = self._determine_fields([key])[0]
        if f not in self.field_data and key not in self.field_data:
            self.get_data(f)

        return self.field_data[f]

    def _determine_fields(self, fields):
        if str(fields) in self.ds._determined_fields:
            return self.ds._determined_fields[str(fields)]
        explicit_fields = []
        for field in iter_fields(fields):
            if field in self._container_fields:
                explicit_fields.append(field)
                continue

            finfo = self.ds._get_field_info(field)
            ftype, fname = finfo.name
            explicit_fields.append((ftype, fname))

        self.ds._determined_fields[str(fields)] = explicit_fields
        return explicit_fields




class YTSelectionContainer(YTDataContainer, abc.ABC):
    _locked = False
    _sort_by = None
    _selector = None
    _current_chunk = None
    _data_source = None
    _dimensionality: int
    _max_level = None
    _min_level = None
    _derived_quantity_chunking = "io"

    def __init__(self, ds, field_parameters, data_source=None):
        super().__init__(ds, field_parameters)
        self._data_source = data_source
        self.quantities = DerivedQuantityCollection(self)

    @property
    def selector(self):
        if self._selector is not None:
            return self._selector
        s_module = getattr(self, "_selector_module", yt.geometry.selection_routines)
        sclass = getattr(s_module, f"{self._type_name}_selector", None)
        self._selector = sclass(self)
        return self._selector

    def _identify_dependencies(self, fields_to_get, spatial=False):
        inspected = 0
        fields_to_get = fields_to_get[:]
        for field in itertools.cycle(fields_to_get):
            if inspected >= len(fields_to_get):
                break
            inspected += 1
            fd = self.ds.field_dependencies.get(
                field, None
            ) or self.ds.field_dependencies.get(field[1], None)
            requested = self._determine_fields(list(set(fd.requested)))
            deps = [d for d in requested if d not in fields_to_get]
            fields_to_get += deps
        return sorted(fields_to_get)

    def get_data(self, fields=None):
        if self._current_chunk is None:
            self.index._identify_base_chunk(self)
        nfields = []
        for field in self._determine_fields(fields):
            # We need to create the field on the raw particle types
            # for particles types (when the field is not directly
            # defined for the derived particle type only)
            finfo = self.ds.field_info[field]

            nfields.append(field)

        fields = nfields
        # Now we collect all our fields
        # Here is where we need to perform a validation step, so that if we
        # have a field requested that we actually *can't* yet get, we put it
        # off until the end.  This prevents double-reading fields that will
        # need to be used in spatial fields later on.
        fields_to_get = []
        # This will be pre-populated with spatial fields
        fields_to_generate = []
        for field in self._determine_fields(fields):
            if field in self.field_data:
                continue
            finfo = self.ds._get_field_info(field)
            finfo.check_available(self)

            fields_to_get.append(field)
        if len(fields_to_get) == 0 and len(fields_to_generate) == 0:
            return
        elif self._locked:
            raise GenerationInProgress(fields)
        # At this point, we want to figure out *all* our dependencies.
        fields_to_get = self._identify_dependencies(fields_to_get, self._spatial)
        # We now split up into readers for the types of fields
        fluids = []
        finfos = {}
        for field_key in fields_to_get:
            finfo = self.ds._get_field_info(field_key)
            finfos[field_key] = finfo
            if field_key not in fluids:
                fluids.append(field_key)
        # The _read method will figure out which fields it needs to get from
        # disk, and return a dict of those fields along with the fields that
        # need to be generated.
        read_fluids, gen_fluids = self.index._read_fluid_fields(
            fluids, self, self._current_chunk
        )
        for f, v in read_fluids.items():
            self.field_data[f] = self.ds.arr(v, units=finfos[f].units)
            self.field_data[f].convert_to_units(finfos[f].output_units)

    @property
    def max_level(self):
        return self.ds.max_level

    @property
    def min_level(self):
        return self.ds.min_level


class YTSelectionContainer3D(YTSelectionContainer):
    """
    Returns an instance of YTSelectionContainer3D, or prepares one.  Usually only
    used as a base class.  Note that *center* is supplied, but only used
    for fields and quantities that require it.
    """

    _key_fields = ["x", "y", "z", "dx", "dy", "dz"]
    _spatial = False
    _num_ghost_zones = 0
    _dimensionality = 3

    def __init__(self, center, ds, field_parameters=None, data_source=None):
        super().__init__(ds, field_parameters, data_source)
        self._set_center(center)
        self.coords = None
        self._grids = None


class YTRegion(YTSelectionContainer3D):
    """A 3D region of data with an arbitrary center.

    Takes an array of three *left_edge* coordinates, three
    *right_edge* coordinates, and a *center* that can be anywhere
    in the domain. If the selected region extends past the edges
    of the domain, no data will be found there, though the
    object's `left_edge` or `right_edge` are not modified.

    Parameters
    ----------
    center : array_like
        The center of the region
    left_edge : array_like
        The left edge of the region
    right_edge : array_like
        The right edge of the region
    """

    _type_name = "region"
    _con_args = ("center", "left_edge", "right_edge")

    def __init__(
        self,
        center,
        left_edge,
        right_edge,
        fields=None,
        ds=None,
        field_parameters=None,
        data_source=None,
    ):
        YTSelectionContainer3D.__init__(self, center, ds, field_parameters, data_source)
        self.left_edge = self.ds.arr(left_edge.copy(), dtype="float64")
        self.right_edge = self.ds.arr(right_edge.copy(), dtype="float64")


def requires_index(attr_name):
    @property
    def ireq(self):
        self.index
        # By now it should have been set
        attr = self.__dict__[attr_name]
        return attr

    @ireq.setter
    def ireq(self, value):
        self.__dict__[attr_name] = value

    return ireq


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
        self.periodicity = periodicity
        self.cell_widths = cell_widths
        self.parameters = {}

    def get_fields(self):
        return self.fields.all_fields


class Dataset(abc.ABC):
    default_fluid_type = "gas"
    default_field = ("gas", "density")
    fluid_types: tuple[FieldType, ...] = ("gas", "deposit", "index")
    geometry: Geometry = Geometry.CARTESIAN
    coordinates = None
    storage_filename = None
    _index_class: type[Index]
    field_units: Optional[dict[AnyFieldKey, Unit]] = None
    derived_field_list = requires_index("derived_field_list")
    fields = requires_index("fields")
    conversion_factors: Optional[dict[str, float]] = None
    # _instantiated represents an instantiation time (since Epoch)
    # the default is a place holder sentinel, falsy value
    _instantiated: float = 0
    _particle_type_counts = None
    _proj_type = "quad_proj"
    _ionization_label_format = "roman_numeral"
    _determined_fields: Optional[dict[str, list[FieldKey]]] = None
    fields_detected = False

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
        """
        Base class for generating new output types.  Principally consists of
        a *filename* and a *dataset_type* which will be passed on to children.
        """
        # We return early and do NOT initialize a second time if this file has
        # already been initialized.
        self.dataset_type = dataset_type
        self.conversion_factors = {}
        self.parameters: dict[str, Any] = {}
        self.region_expression = self.r = RegionExpression(self)
        self.field_units = self.field_units or {}
        self._determined_fields = {}
        self.units_override = units_override
        self.default_species_fields = default_species_fields

        self._input_filename: str = os.fspath(filename)

        # to get the timing right, do this before the heavy lifting
        self._instantiated = time.time()

        self.no_cgs_equiv_length = False
        self._create_unit_registry(unit_system)

        self._parse_parameter_file()
        self.set_units()
        self._assign_unit_system(unit_system)
        self._setup_coordinate_handler(axis_order)
        self._set_derived_attrs()
        self._setup_classes()

    @property
    def basename(self):
        return os.path.basename(self.filename)

    @property
    def directory(self):
        return os.path.dirname(self.filename)

    @property
    def periodicity(self):
        return self._periodicity

    def _set_derived_attrs(self):
        self.domain_center = 0.5 * (self.domain_right_edge + self.domain_left_edge)
        self.domain_width = self.domain_right_edge - self.domain_left_edge
        if not isinstance(self.current_time, YTQuantity):
            self.current_time = self.quan(self.current_time, "code_time")
        for attr in ("center", "width", "left_edge", "right_edge"):
            n = f"domain_{attr}"
            v = getattr(self, n)
            if not isinstance(v, YTArray) and v is not None:
                # Note that we don't add on _ipython_display_ here because
                # everything is stored inside a MutableAttribute.
                v = self.arr(v, "code_length")
                setattr(self, n, v)

    def _hash(self):
        s = f"{self.basename};{self.current_time};{123456789}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    _instantiated_index = None

    @property
    def index(self):
        if self._instantiated_index is None:
            self._instantiated_index = self._index_class(
                self, dataset_type=self.dataset_type
            )
            # Now we do things that we need an instantiated index for
            # ...first off, we create our field_info now.
            oldsettings = np.geterr()
            np.seterr(all="ignore")
            self.create_field_info()
            np.seterr(**oldsettings)
        return self._instantiated_index

    @property
    def field_list(self):
        return self.index.field_list

    def create_field_info(self):
        self.field_dependencies = {}
        self.derived_field_list = []
        self.field_info = self._field_info_class(self, self.field_list)
        self.coordinates.setup_fields(self.field_info)
        self.field_info.setup_fluid_fields()
        self.field_info.setup_fluid_index_fields()

        self.field_info.load_all_plugins(self.default_fluid_type)
        deps, unloaded = self.field_info.check_derived_fields()
        self.field_dependencies.update(deps)
        self.fields = FieldTypeContainer(self)
        self.index.field_list = sorted(self.field_list)
        # Now that we've detected the fields, set this flag so that
        # deprecated fields will be logged if they are used
        self.fields_detected = True

    def _setup_coordinate_handler(self, axis_order: Optional[AxisOrder]) -> None:
        self.coordinates = CartesianCoordinateHandler(self, ordering=axis_order)

    def _get_field_info(
        self,
        field: Union[FieldKey, ImplicitFieldKey, DerivedField],
        /,
    ) -> DerivedField:
        field_info, candidates = self._get_field_info_helper(field)
        return field_info

    def _get_field_info_helper(
        self,
        field: Union[FieldKey, ImplicitFieldKey, DerivedField],
        /,
    ) -> tuple[DerivedField, list[FieldKey]]:
        self.index

        ftype: str
        fname: str
        if isinstance(field, tuple) and len(field) == 2:
            ftype, fname = field
        else:
            raise TypeError(field)

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

    # Now all the object related stuff
    def all_data(self, **kwargs):
        """
        all_data is a wrapper to the Region object for creating a region
        which covers the entire simulation domain.
        """
        self.index
        c = (self.domain_right_edge + self.domain_left_edge) / 2.0
        return self.region(c, self.domain_left_edge, self.domain_right_edge, **kwargs)

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
        mag_unit: Optional[unyt_quantity] = getattr(self, "magnetic_unit", None)
        mag_dims = mag_unit.units.dimensions.free_symbols

        if unit_system != "code":
            # if the unit system is known, we can check if it
            # has a "current_mks" unit
            us = unit_system_registry[str(unit_system).lower()]
            mks_system = us.base_units[current_mks] is not None
        elif mag_dims and current_mks in mag_dims:
            # if we're using the not-yet defined code unit system,
            # then we check if the magnetic field unit has a SI
            # current dimension in it
            mks_system = True

        current_mks_unit = "A" if mks_system else None
        us = create_code_unit_system(
            self.unit_registry, current_mks_unit=current_mks_unit
        )
        if unit_system != "code":
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

    _units = None
    _unit_system_id = None

    @property
    def units(self):
        current_uid = self.unit_registry.unit_system_id
        if self._units is not None and self._unit_system_id == current_uid:
            return self._units
        self._unit_system_id = current_uid
        self._units = UnitContainer(self.unit_registry)
        return self._units

    _arr = None

    @property
    def arr(self):
        """Converts an array into a :class:`yt.units.yt_array.YTArray`

        The returned YTArray will be dimensionless by default, but can be
        cast to arbitrary units using the ``units`` keyword argument.

        Parameters
        ----------

        input_array : Iterable
            A tuple, list, or array to attach units to
        units: String unit specification, unit symbol or astropy object
            The units of the array. Powers must be specified using python syntax
            (cm**3, not cm^3).
        input_units : Deprecated in favor of 'units'
        dtype : string or NumPy dtype object
            The dtype of the returned array data

        Examples
        --------

        >>> import yt
        >>> import numpy as np
        >>> ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")
        >>> a = ds.arr([1, 2, 3], "cm")
        >>> b = ds.arr([4, 5, 6], "m")
        >>> a + b
        YTArray([ 401.,  502.,  603.]) cm
        >>> b + a
        YTArray([ 4.01,  5.02,  6.03]) m

        Arrays returned by this function know about the dataset's unit system

        >>> a = ds.arr(np.ones(5), "code_length")
        >>> a.in_units("Mpccm/h")
        YTArray([ 1.00010449,  1.00010449,  1.00010449,  1.00010449,
                 1.00010449]) Mpc

        """

        if self._arr is not None:
            return self._arr
        self._arr = functools.partial(YTArray, registry=self.unit_registry)
        return self._arr

    _quan = None

    @property
    def quan(self):
        """Converts an scalar into a :class:`yt.units.yt_array.YTQuantity`

        The returned YTQuantity will be dimensionless by default, but can be
        cast to arbitrary units using the ``units`` keyword argument.

        Parameters
        ----------

        input_scalar : an integer or floating point scalar
            The scalar to attach units to
        units: String unit specification, unit symbol or astropy object
            The units of the quantity. Powers must be specified using python
            syntax (cm**3, not cm^3).
        input_units : Deprecated in favor of 'units'
        dtype : string or NumPy dtype object
            The dtype of the array data.

        Examples
        --------

        >>> import yt
        >>> ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")

        >>> a = ds.quan(1, "cm")
        >>> b = ds.quan(2, "m")
        >>> a + b
        201.0 cm
        >>> b + a
        2.01 m

        Quantities created this way automatically know about the unit system
        of the dataset.

        >>> a = ds.quan(5, "code_length")
        >>> a.in_cgs()
        1.543e+25 cm

        """

        if self._quan is not None:
            return self._quan
        self._quan = functools.partial(YTQuantity, registry=self.unit_registry)
        return self._quan


class FieldInfoContainer(UserDict):
    """
    This is a generic field container.  It contains a list of potential derived
    fields, all of which know how to act on a data object and return a value.
    This object handles converting units as well as validating the availability
    of a given field.

    """

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
        self.field_aliases: dict[FieldKey, FieldKey] = {}
        self.species_names: list[FieldName] = []
        self.setup_fluid_aliases()

    def setup_fluid_index_fields(self):
        # Now we get all our index types and set up aliases to them
        index_fields = {f for _, f in self if _ == "index"}
        for ftype in self.ds.fluid_types:
            if ftype in ("index", "deposit"):
                continue
            for f in index_fields:
                self.alias((ftype, f), ("index", f))

    def setup_fluid_aliases(self, ftype: FieldType = "gas") -> None:
        known_other_fields = dict(self.known_other_fields)

        for field in sorted(self.field_list):
            units, aliases, display_name = known_other_fields.get(field[1], None)

            # We allow field_units to override this.  First we check if the
            # field *name* is in there, then the field *tuple*.
            units = self.ds.field_units.get(field[1], units)
            units = self.ds.field_units.get(field, units)
            self.add_output_field(
                field, sampling_type="cell", units=units, display_name=display_name
            )
            for alias in aliases:
                self.alias((ftype, alias), field)

    def add_field(
        self,
        name: FieldKey,
        function: Callable,
        sampling_type: str,
        *,
        alias: Optional[DerivedField] = None,
        force_override: bool = False,
        **kwargs,
    ) -> None:
        """
        Add a new field, along with supplemental metadata, to the list of
        available fields.  This respects a number of arguments, all of which
        are passed on to the constructor for
        :class:`~yt.data_objects.api.DerivedField`.

        Parameters
        ----------

        name : tuple[str, str]
           field (or particle) type, field name
        function : callable
           A function handle that defines the field.  Should accept
           arguments (field, data)
        sampling_type: str
           "cell" or "particle" or "local"
        force_override: bool
           If False (default), an error will be raised if a field of the same name already exists.
        alias: DerivedField (optional):
           existing field to be aliased
        units : str
           A plain text string encoding the unit.  Powers must be in
           python syntax (** instead of ^). If set to "auto" the units
           will be inferred from the return value of the field function.
        take_log : bool
           Describes whether the field should be logged
        validators : list
           A list of :class:`FieldValidator` objects
        vector_field : bool
           Describes the dimensionality of the field.  Currently unused.
        display_name : str
           A name used in the plots

        """
        # Handle the case where the field has already been added.
        if not force_override and name in self:
            return

        kwargs.setdefault("ds", self.ds)

        if (
            not isinstance(name, str)
            and obj_length(name) == 2
            and all(isinstance(e, str) for e in name)
        ):
            self[name] = DerivedField(
                name, sampling_type, function, alias=alias, **kwargs
            )
        else:
            raise ValueError(f"Expected name to be a tuple[str, str], got {name}")

    def load_all_plugins(self, ftype: Optional[str] = "gas") -> None:
        loaded = []
        for n in sorted(field_plugins):
            loaded += self.load_plugin(n, ftype)
        self.find_dependencies(loaded)

    def load_plugin(
        self,
        plugin_name: FunctionName,
        ftype: FieldType = "gas",
        skip_check: bool = False,
    ):
        f = field_plugins[plugin_name]
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
        """
        Alias one field to another field.

        Parameters
        ----------
        alias_name : tuple[str, str]
            The new field name.
        original_name : tuple[str, str]
            The field to be aliased.
        units : str
           A plain text string encoding the unit.  Powers must be in
           python syntax (** instead of ^). If set to "auto" the units
           will be inferred from the return value of the field function.
        deprecate : tuple[str, str | None] | None
            If this is set, then the tuple contains two string version
            numbers: the first marking the version when the field was
            deprecated, and the second marking when the field will be
            removed.
        """
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
        blacklist = ()
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
                # fd: field detector
                fd = fi.get_dependencies(ds=self.ds)
            except blacklist as err:
                print(f"{err.__class__} raised for field {field}")
                raise SystemExit(1) from err
            except (*whitelist, *greylist) as e:
                if field in self._show_field_errors:
                    raise
                if not isinstance(e, YTFieldNotFound):
                    # if we're doing field tests, raise an error
                    # see yt.fields.tests.test_fields
                    if hasattr(self.ds, "_field_test_dataset"):
                        raise
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
        self._set_linear_fields()
        return deps, unavailable

    def _set_linear_fields(self):
        """
        Sets which fields use linear as their default scaling in Profiles and
        PhasePlots. Default for all fields is set to log, so this sets which
        are linear.  For now, set linear to geometric fields: position and
        velocity coordinates.
        """
        non_log_prefixes = ("", "velocity_", "particle_position_", "particle_velocity_")
        coords = ("x", "y", "z")
        non_log_fields = [
            prefix + coord for prefix in non_log_prefixes for coord in coords
        ]
        for field in self.ds.derived_field_list:
            if field[1] in non_log_fields:
                self[field].take_log = False


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
    _index_class = StreamHierarchy
    _field_info_class = StreamFieldInfo
    _dataset_type = "stream"

    def __init__(self, *, stream_handler):
        self.fluid_types += ("stream",)
        self.stream_handler = stream_handler
        self.filename = self.stream_handler.name
        Dataset.__init__(
            self,
            filename=f"InMemoryParameterFile_{uuid.uuid4().hex}",
            dataset_type=self._dataset_type,
            unit_system="cgs",
        )

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
