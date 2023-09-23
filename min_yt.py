from __future__ import annotations

import abc
import functools
import hashlib
import itertools
import os
import re
import time
import uuid
import weakref
from collections import UserDict, defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from functools import cached_property
from importlib.util import find_spec
from itertools import chain
from typing import Any, Literal, Optional, Union

import numpy as np
import unyt as un
import yt.geometry.selection_routines
from more_itertools import always_iterable
from sympy import Symbol
from typing_extensions import assert_never
from unyt import Unit, UnitSystem, unyt_quantity
from unyt.exceptions import UnitConversionError, UnitParseError
from yt._typing import AnyFieldKey, FieldKey, FieldName, FieldType, KnownFieldsT
from yt.config import ytcfg
from yt.data_objects.derived_quantities import DerivedQuantityCollection
from yt.data_objects.field_data import YTFieldData
from yt.data_objects.profiles import create_profile
from yt.data_objects.region_expression import RegionExpression
from yt.data_objects.static_output import _cached_datasets, _ds_store
from yt.data_objects.unions import ParticleUnion
from yt.fields.derived_field import (
    DerivedField,
    NullFunc,
    TranslationFunc,
    ValidateSpatial,
)
from yt.fields.field_exceptions import NeedsConfiguration, NeedsGridType
from yt.fields.field_functions import validate_field_function
from yt.fields.field_plugin_registry import FunctionName, field_plugins
from yt.fields.field_type_container import FieldTypeContainer
from yt.fields.fluid_fields import setup_gradient_fields
from yt.fields.particle_fields import (
    add_union_field,
    particle_deposition_functions,
    particle_scalar_functions,
    particle_vector_functions,
    sph_whitelist_fields,
    standard_particle_fields,
)
from yt.frontends.stream.api import StreamHierarchy
from yt.frontends.ytdata.utilities import save_as_dataset
from yt.funcs import (
    get_output_filename,
    iter_fields,
    obj_length,
)
from yt.geometry.api import Geometry
from yt.geometry.coordinates.api import (
    CartesianCoordinateHandler
)
from yt.geometry.geometry_handler import Index
from yt.geometry.selection_routines import compose_selector
from yt.units import UnitContainer, _wrap_display_ytarray, dimensions
from yt.units._numpy_wrapper_functions import uconcatenate
from yt.units.dimensions import (
    current_mks,
    dimensionless,  # type: ignore
)
from yt.units.unit_object import define_unit
from yt.units.unit_registry import UnitRegistry
from yt.units.unit_systems import (
    create_code_unit_system,
    unit_system_registry,
)
from yt.units.yt_array import YTArray, YTQuantity
from yt.utilities.amr_kdtree.api import AMRKDTree
from yt.utilities.cosmology import Cosmology
from yt.utilities.exceptions import (
    GenerationInProgress,
    YTBooleanObjectError,
    YTBooleanObjectsWrongDataset,
    YTCoordinateNotImplemented,
    YTCouldNotGenerateField,
    YTDataSelectorNotImplemented,
    YTDimensionalityError,
    YTDomainOverflow,
    YTException,
    YTFieldNotFound,
    YTFieldNotParseable,
    YTFieldUnitError,
    YTFieldUnitParseError,
    YTNonIndexedDataContainer,
    YTSpatialFieldUnitError,
)
from yt.utilities.object_registries import data_object_registry
from yt.utilities.parallel_tools.parallel_analysis_interface import (
    ParallelAnalysisInterface,
)
from yt.utilities.parameter_file_storage import NoParameterShelf


def sanitize_weight_field(weight):
    if weight is None:
        weight_field = ("index", "ones")
    else:
        weight_field = weight
    return weight_field


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

        self.ds: "Dataset"
        if ds is not None:
            self.ds = ds
        else:
            if not hasattr(self, "ds"):
                raise RuntimeError(
                    "Error: ds must be set either through class type "
                    "or parameter to the constructor"
                )

        self._current_fluid_type = self.ds.default_fluid_type
        self.ds.objects.append(weakref.proxy(self))
        self.field_data = YTFieldData()
        if self.ds.unit_system.has_current_mks:
            mag_unit = "T"
        else:
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
        for key, val in field_parameters.items():
            self.set_field_parameter(key, val)

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
            self.set_field_parameter(k, v)

    def _is_default_field_parameter(self, parameter):
        if parameter not in self._default_field_parameters:
            return False
        return (
            self._default_field_parameters[parameter]
            is self.field_parameters[parameter]
        )

    def apply_units(self, arr, units):
        try:
            arr.units.registry = self.ds.unit_registry
            return arr.to(units)
        except AttributeError:
            return self.ds.arr(arr, units=units)

    def _first_matching_field(self, field: FieldName) -> FieldKey:
        for ftype, fname in self.ds.derived_field_list:
            if fname == field:
                return (ftype, fname)

        raise YTFieldNotFound(field, self.ds)

    def _set_center(self, center):
        if center is None:
            self.center = None
            return
        else:
            self.center = center
            self.set_field_parameter("center", self.center)

    def get_field_parameter(self, name, default=None):
        """
        This is typically only used by derived field functions, but
        it returns parameters used to generate fields.
        """
        if name in self.field_parameters:
            return self.field_parameters[name]
        else:
            return default

    def set_field_parameter(self, name, val):
        """
        Here we set up dictionaries that get passed up and down and ultimately
        to derived fields.
        """
        self.field_parameters[name] = val

    def has_field_parameter(self, name):
        """
        Checks if a field parameter is set.
        """
        return name in self.field_parameters

    def clear_data(self):
        """
        Clears out all data from the YTDataContainer instance, freeing memory.
        """
        self.field_data.clear()

    def __getitem__(self, key):
        """
        Returns a single field.  Will add if necessary.
        """
        f = self._determine_fields([key])[0]
        if f not in self.field_data and key not in self.field_data:
            if f in self._container_fields:
                self.field_data[f] = self.ds.arr(self._generate_container_field(f))
                return self.field_data[f]
            else:
                self.get_data(f)
        # fi.units is the unit expression string. We depend on the registry
        # hanging off the dataset to define this unit object.
        # Note that this is less succinct so that we can account for the case
        # when there are, for example, no elements in the object.
        try:
            rv = self.field_data[f]
        except KeyError:
            fi = self.ds._get_field_info(f)
            rv = self.ds.arr(self.field_data[key], fi.units)
        return rv

    def __setitem__(self, key, val):
        """
        Sets a field to be some other value.
        """
        self.field_data[key] = val

    def __delitem__(self, key):
        """
        Deletes a field
        """
        if key not in self.field_data:
            key = self._determine_fields(key)[0]
        del self.field_data[key]

    def _generate_field(self, field):
        ftype, fname = field
        finfo = self.ds._get_field_info(field)
        with self._field_type_state(ftype, finfo):
            if fname in self._container_fields:
                tr = self._generate_container_field(field)
            if finfo.sampling_type == "particle":
                tr = self._generate_particle_field(field)
            else:
                tr = self._generate_fluid_field(field)
            if tr is None:
                raise YTCouldNotGenerateField(field, self.ds)
            return tr

    def _generate_fluid_field(self, field):
        # First we check the validator
        finfo = self.ds._get_field_info(field)
        if self._current_chunk is None or self._current_chunk.chunk_type != "spatial":
            gen_obj = self
        else:
            gen_obj = self._current_chunk.objs[0]
            gen_obj.field_parameters = self.field_parameters
        try:
            finfo.check_available(gen_obj)
        except NeedsGridType as ngt_exception:
            rv = self._generate_spatial_fluid(field, ngt_exception.ghost_zones)
        else:
            rv = finfo(gen_obj)
        return rv

    def _generate_spatial_fluid(self, field, ngz):
        finfo = self.ds._get_field_info(field)
        if finfo.units is None:
            raise YTSpatialFieldUnitError(field)
        units = finfo.units
        try:
            rv = self.ds.arr(np.zeros(self.ires.size, dtype="float64"), units)
            accumulate = False
        except YTNonIndexedDataContainer:
            # In this case, we'll generate many tiny arrays of unknown size and
            # then concatenate them.
            outputs = []
            accumulate = True
        ind = 0
        if ngz == 0:
            deps = self._identify_dependencies([field], spatial=True)
            deps = self._determine_fields(deps)
            for _io_chunk in self.chunks([], "io", cache=False):
                for _chunk in self.chunks([], "spatial", ngz=0, preload_fields=deps):
                    o = self._current_chunk.objs[0]
                    if accumulate:
                        rv = self.ds.arr(np.empty(o.ires.size, dtype="float64"), units)
                        outputs.append(rv)
                        ind = 0  # Does this work with mesh?
                    with o._activate_cache():
                        ind += o.select(
                            self.selector, source=self[field], dest=rv, offset=ind
                        )
        else:
            chunks = self.index._chunk(self, "spatial", ngz=ngz)
            for chunk in chunks:
                with self._chunked_read(chunk):
                    gz = self._current_chunk.objs[0]
                    gz.field_parameters = self.field_parameters
                    wogz = gz._base_grid
                    if accumulate:
                        rv = self.ds.arr(
                            np.empty(wogz.ires.size, dtype="float64"), units
                        )
                        outputs.append(rv)
                    ind += wogz.select(
                        self.selector,
                        source=gz[field][ngz:-ngz, ngz:-ngz, ngz:-ngz],
                        dest=rv,
                        offset=ind,
                    )
        if accumulate:
            rv = uconcatenate(outputs)
        return rv


    def _generate_container_field(self, field):
        raise NotImplementedError

    def _parameter_iterate(self, seq):
        for obj in seq:
            old_fp = obj.field_parameters
            obj.field_parameters = self.field_parameters
            yield obj
            obj.field_parameters = old_fp


    # Numpy-like Operations
    def argmax(self, field, axis=None):
        r"""Return the values at which the field is maximized.

        This will, in a parallel-aware fashion, find the maximum value and then
        return to you the values at that maximum location that are requested
        for "axis".  By default it will return the spatial positions (in the
        natural coordinate system), but it can be any field

        Parameters
        ----------
        field : string or tuple field name
            The field to maximize.
        axis : string or list of strings, optional
            If supplied, the fields to sample along; if not supplied, defaults
            to the coordinate fields.  This can be the name of the coordinate
            fields (i.e., 'x', 'y', 'z') or a list of fields, but cannot be 0,
            1, 2.

        Returns
        -------
        A list of YTQuantities as specified by the axis argument.

        Examples
        --------

        >>> temp_at_max_rho = reg.argmax(
        ...     ("gas", "density"), axis=("gas", "temperature")
        ... )
        >>> max_rho_xyz = reg.argmax(("gas", "density"))
        >>> t_mrho, v_mrho = reg.argmax(
        ...     ("gas", "density"),
        ...     axis=[("gas", "temperature"), ("gas", "velocity_magnitude")],
        ... )
        >>> x, y, z = reg.argmax(("gas", "density"))

        """
        if axis is None:
            mv, pos0, pos1, pos2 = self.quantities.max_location(field)
            return pos0, pos1, pos2
        if isinstance(axis, str):
            axis = [axis]
        rv = self.quantities.sample_at_max_field_values(field, axis)
        if len(rv) == 2:
            return rv[1]
        return rv[1:]

    def argmin(self, field, axis=None):
        r"""Return the values at which the field is minimized.

        This will, in a parallel-aware fashion, find the minimum value and then
        return to you the values at that minimum location that are requested
        for "axis".  By default it will return the spatial positions (in the
        natural coordinate system), but it can be any field

        Parameters
        ----------
        field : string or tuple field name
            The field to minimize.
        axis : string or list of strings, optional
            If supplied, the fields to sample along; if not supplied, defaults
            to the coordinate fields.  This can be the name of the coordinate
            fields (i.e., 'x', 'y', 'z') or a list of fields, but cannot be 0,
            1, 2.

        Returns
        -------
        A list of YTQuantities as specified by the axis argument.

        Examples
        --------

        >>> temp_at_min_rho = reg.argmin(
        ...     ("gas", "density"), axis=("gas", "temperature")
        ... )
        >>> min_rho_xyz = reg.argmin(("gas", "density"))
        >>> t_mrho, v_mrho = reg.argmin(
        ...     ("gas", "density"),
        ...     axis=[("gas", "temperature"), ("gas", "velocity_magnitude")],
        ... )
        >>> x, y, z = reg.argmin(("gas", "density"))

        """
        if axis is None:
            mv, pos0, pos1, pos2 = self.quantities.min_location(field)
            return pos0, pos1, pos2
        if isinstance(axis, str):
            axis = [axis]
        rv = self.quantities.sample_at_min_field_values(field, axis)
        if len(rv) == 2:
            return rv[1]
        return rv[1:]

    def _compute_extrema(self, field):
        if self._extrema_cache is None:
            self._extrema_cache = {}
        if field not in self._extrema_cache:
            # Note we still need to call extrema for each field, as of right
            # now
            mi, ma = self.quantities.extrema(field)
            self._extrema_cache[field] = (mi, ma)
        return self._extrema_cache[field]

    _extrema_cache = None

    def max(self, field, axis=None):
        r"""Compute the maximum of a field, optionally along an axis.

        This will, in a parallel-aware fashion, compute the maximum of the
        given field.  Supplying an axis will result in a return value of a
        YTProjection, with method 'max' for maximum intensity.  If the max has
        already been requested, it will use the cached extrema value.

        Parameters
        ----------
        field : string or tuple field name
            The field to maximize.
        axis : string, optional
            If supplied, the axis to project the maximum along.

        Returns
        -------
        Either a scalar or a YTProjection.

        Examples
        --------

        >>> max_temp = reg.max(("gas", "temperature"))
        >>> max_temp_proj = reg.max(("gas", "temperature"), axis=("index", "x"))
        """
        if axis is None:
            rv = tuple(self._compute_extrema(f)[1] for f in iter_fields(field))
            if len(rv) == 1:
                return rv[0]
            return rv
        elif axis in self.ds.coordinates.axis_name:
            return self.ds.proj(field, axis, data_source=self, method="max")
        else:
            raise NotImplementedError(f"Unknown axis {axis}")

    def min(self, field, axis=None):
        r"""Compute the minimum of a field.

        This will, in a parallel-aware fashion, compute the minimum of the
        given field. Supplying an axis will result in a return value of a
        YTProjection, with method 'min' for minimum intensity.  If the min
        has already been requested, it will use the cached extrema value.

        Parameters
        ----------
        field : string or tuple field name
            The field to minimize.
        axis : string, optional
            If supplied, the axis to compute the minimum along.

        Returns
        -------
        Either a scalar or a YTProjection.

        Examples
        --------

        >>> min_temp = reg.min(("gas", "temperature"))
        >>> min_temp_proj = reg.min(("gas", "temperature"), axis=("index", "x"))
        """
        if axis is None:
            rv = tuple(self._compute_extrema(f)[0] for f in iter_fields(field))
            if len(rv) == 1:
                return rv[0]
            return rv
        elif axis in self.ds.coordinates.axis_name:
            return self.ds.proj(field, axis, data_source=self, method="min")
        else:
            raise NotImplementedError(f"Unknown axis {axis}")

    def std(self, field, axis=None, weight=None):
        """Compute the standard deviation of a field, optionally along
        an axis, with a weight.

        This will, in a parallel-ware fashion, compute the standard
        deviation of the given field. If an axis is supplied, it
        will return a projection, where the weight is also supplied.

        By default the weight field will be "ones" or "particle_ones",
        depending on the field, resulting in an unweighted standard
        deviation.

        Parameters
        ----------
        field : string or tuple field name
            The field to calculate the standard deviation of
        axis : string, optional
            If supplied, the axis to compute the standard deviation
            along (i.e., to project along)
        weight : string, optional
            The field to use as a weight.

        Returns
        -------
        Scalar or YTProjection.
        """
        weight_field = sanitize_weight_field(weight)
        if axis in self.ds.coordinates.axis_name:
            r = self.ds.proj(
                field, axis, data_source=self, weight_field=weight_field, moment=2
            )
        elif axis is None:
            r = self.quantities.weighted_standard_deviation(field, weight_field)[0]
        else:
            raise NotImplementedError(f"Unknown axis {axis}")
        return r

    def ptp(self, field):
        r"""Compute the range of values (maximum - minimum) of a field.

        This will, in a parallel-aware fashion, compute the "peak-to-peak" of
        the given field.

        Parameters
        ----------
        field : string or tuple field name
            The field to average.

        Returns
        -------
        Scalar

        Examples
        --------

        >>> rho_range = reg.ptp(("gas", "density"))
        """
        ex = self._compute_extrema(field)
        return ex[1] - ex[0]

    def profile(
        self,
        bin_fields,
        fields,
        n_bins=64,
        extrema=None,
        logs=None,
        units=None,
        weight_field=("gas", "mass"),
        accumulation=False,
        fractional=False,
        deposition="ngp",
    ):
        r"""
        Create a 1, 2, or 3D profile object from this data_source.

        The dimensionality of the profile object is chosen by the number of
        fields given in the bin_fields argument.  This simply calls
        :func:`yt.data_objects.profiles.create_profile`.

        Parameters
        ----------
        bin_fields : list of strings
            List of the binning fields for profiling.
        fields : list of strings
            The fields to be profiled.
        n_bins : int or list of ints
            The number of bins in each dimension.  If None, 64 bins for
            each bin are used for each bin field.
            Default: 64.
        extrema : dict of min, max tuples
            Minimum and maximum values of the bin_fields for the profiles.
            The keys correspond to the field names. Defaults to the extrema
            of the bin_fields of the dataset. If a units dict is provided, extrema
            are understood to be in the units specified in the dictionary.
        logs : dict of boolean values
            Whether or not to log the bin_fields for the profiles.
            The keys correspond to the field names. Defaults to the take_log
            attribute of the field.
        units : dict of strings
            The units of the fields in the profiles, including the bin_fields.
        weight_field : str or tuple field identifier
            The weight field for computing weighted average for the profile
            values.  If None, the profile values are sums of the data in
            each bin.
        accumulation : bool or list of bools
            If True, the profile values for a bin n are the cumulative sum of
            all the values from bin 0 to n.  If -True, the sum is reversed so
            that the value for bin n is the cumulative sum from bin N (total bins)
            to n.  If the profile is 2D or 3D, a list of values can be given to
            control the summation in each dimension independently.
            Default: False.
        fractional : If True the profile values are divided by the sum of all
            the profile data such that the profile represents a probability
            distribution function.
        deposition : Controls the type of deposition used for ParticlePhasePlots.
            Valid choices are 'ngp' and 'cic'. Default is 'ngp'. This parameter is
            ignored the if the input fields are not of particle type.


        Examples
        --------

        Create a 1d profile.  Access bin field from profile.x and field
        data from profile[<field_name>].

        >>> ds = load("DD0046/DD0046")
        >>> ad = ds.all_data()
        >>> profile = ad.profile(
        ...     ad,
        ...     [("gas", "density")],
        ...     [("gas", "temperature"), ("gas", "velocity_x")],
        ... )
        >>> print(profile.x)
        >>> print(profile["gas", "temperature"])
        >>> plot = profile.plot()
        """
        p = create_profile(
            self,
            bin_fields,
            fields,
            n_bins,
            extrema,
            logs,
            units,
            weight_field,
            accumulation,
            fractional,
            deposition,
        )
        return p

    def mean(self, field, axis=None, weight=None):
        r"""Compute the mean of a field, optionally along an axis, with a
        weight.

        This will, in a parallel-aware fashion, compute the mean of the
        given field.  If an axis is supplied, it will return a projection,
        where the weight is also supplied.  By default the weight field will be
        "ones" or "particle_ones", depending on the field being averaged,
        resulting in an unweighted average.

        Parameters
        ----------
        field : string or tuple field name
            The field to average.
        axis : string, optional
            If supplied, the axis to compute the mean along (i.e., to project
            along)
        weight : string, optional
            The field to use as a weight.

        Returns
        -------
        Scalar or YTProjection.

        Examples
        --------

        >>> avg_rho = reg.mean(("gas", "density"), weight="cell_volume")
        >>> rho_weighted_T = reg.mean(
        ...     ("gas", "temperature"), axis=("index", "y"), weight=("gas", "density")
        ... )
        """
        weight_field = sanitize_weight_field(weight)
        if axis in self.ds.coordinates.axis_name:
            r = self.ds.proj(field, axis, data_source=self, weight_field=weight_field)
        elif axis is None:
            r = self.quantities.weighted_average_quantity(field, weight_field)
        else:
            raise NotImplementedError(f"Unknown axis {axis}")
        return r

    def sum(self, field, axis=None):
        r"""Compute the sum of a field, optionally along an axis.

        This will, in a parallel-aware fashion, compute the sum of the given
        field.  If an axis is specified, it will return a projection (using
        method type "sum", which does not take into account path length) along
        that axis.

        Parameters
        ----------
        field : string or tuple field name
            The field to sum.
        axis : string, optional
            If supplied, the axis to sum along.

        Returns
        -------
        Either a scalar or a YTProjection.

        Examples
        --------

        >>> total_vol = reg.sum("cell_volume")
        >>> cell_count = reg.sum(("index", "ones"), axis=("index", "x"))
        """
        # Because we're using ``sum`` to specifically mean a sum or a
        # projection with the method="sum", we do not utilize the ``mean``
        # function.
        if axis in self.ds.coordinates.axis_name:
            with self._field_parameter_state({"axis": axis}):
                r = self.ds.proj(field, axis, data_source=self, method="sum")
        elif axis is None:
            r = self.quantities.total_quantity(field)
        else:
            raise NotImplementedError(f"Unknown axis {axis}")
        return r

    def integrate(self, field, weight=None, axis=None, *, moment=1):
        r"""Compute the integral (projection) of a field along an axis.

        This projects a field along an axis.

        Parameters
        ----------
        field : string or tuple field name
            The field to project.
        weight : string or tuple field name
            The field to weight the projection by
        axis : string
            The axis to project along.
        moment : integer, optional
            for a weighted projection, moment = 1 (the default) corresponds to a
            weighted average. moment = 2 corresponds to a weighted standard
            deviation.

        Returns
        -------
        YTProjection

        Examples
        --------

        >>> column_density = reg.integrate(("gas", "density"), axis=("index", "z"))
        """
        if weight is not None:
            weight_field = sanitize_weight_field(weight)
        else:
            weight_field = None
        if axis in self.ds.coordinates.axis_name:
            r = self.ds.proj(
                field, axis, data_source=self, weight_field=weight_field, moment=moment
            )
        else:
            raise NotImplementedError(f"Unknown axis {axis}")
        return r

    @property
    def _hash(self):
        s = f"{self}"
        try:
            import hashlib

            return hashlib.md5(s.encode("utf-8")).hexdigest()
        except ImportError:
            return s

    def __reduce__(self):
        args = tuple(
            [self.ds._hash(), self._type_name]
            + [getattr(self, n) for n in self._con_args]
            + [self.field_parameters]
        )
        return (_reconstruct_object, args)

    def clone(self):
        r"""Clone a data object.

        This will make a duplicate of a data object; note that the
        `field_parameters` may not necessarily be deeply-copied.  If you modify
        the field parameters in-place, it may or may not be shared between the
        objects, depending on the type of object that that particular field
        parameter is.

        Notes
        -----
        One use case for this is to have multiple identical data objects that
        are being chunked over in different orders.

        Examples
        --------

        >>> ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")
        >>> sp = ds.sphere("c", 0.1)
        >>> sp_clone = sp.clone()
        >>> sp[("gas", "density")]
        >>> print(sp.field_data.keys())
        [("gas", "density")]
        >>> print(sp_clone.field_data.keys())
        []
        """
        args = self.__reduce__()
        return args[0](self.ds, *args[1][1:])

    def __repr__(self):
        # We'll do this the slow way to be clear what's going on
        s = f"{self.__class__.__name__} ({self.ds}): "
        for i in self._con_args:
            try:
                s += ", {}={}".format(
                    i,
                    getattr(self, i).in_base(unit_system=self.ds.unit_system),
                )
            except AttributeError:
                s += f", {i}={getattr(self, i)}"
        return s

    @contextmanager
    def _field_parameter_state(self, field_parameters):
        # What we're doing here is making a copy of the incoming field
        # parameters, and then updating it with our own.  This means that we'll
        # be using our own center, if set, rather than the supplied one.  But
        # it also means that any additionally set values can override it.
        old_field_parameters = self.field_parameters
        new_field_parameters = field_parameters.copy()
        new_field_parameters.update(old_field_parameters)
        self.field_parameters = new_field_parameters
        yield
        self.field_parameters = old_field_parameters

    @contextmanager
    def _field_type_state(self, ftype, finfo, obj=None):
        if obj is None:
            obj = self
        old_fluid_type = obj._current_fluid_type
        obj._current_fluid_type = ftype
        yield
        obj._current_fluid_type = old_fluid_type

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
            # really ugly check to ensure that this field really does exist somewhere,
            # in some naming convention, before returning it as a possible field type
            if (
                (ftype, fname) not in self.ds.field_info
                and (ftype, fname) not in self.ds.field_list
                and fname not in self.ds.field_list
                and (ftype, fname) not in self.ds.derived_field_list
                and fname not in self.ds.derived_field_list
                and (ftype, fname) not in self._container_fields
            ):
                raise YTFieldNotFound((ftype, fname), self.ds)

            explicit_fields.append((ftype, fname))

        self.ds._determined_fields[str(fields)] = explicit_fields
        return explicit_fields

    _tree = None

    @property
    def tiles(self):
        if self._tree is not None:
            return self._tree
        self._tree = AMRKDTree(self.ds, data_source=self)
        return self._tree

    @property
    def blocks(self):
        for _io_chunk in self.chunks([], "io"):
            for _chunk in self.chunks([], "spatial", ngz=0):
                # For grids this will be a grid object, and for octrees it will
                # be an OctreeSubset.  Note that we delegate to the sub-object.
                o = self._current_chunk.objs[0]
                cache_fp = o.field_parameters.copy()
                o.field_parameters.update(self.field_parameters)
                for b, m in o.select_blocks(self.selector):
                    if m is None:
                        continue
                    yield b, m
                o.field_parameters = cache_fp


class YTSelectionContainer(YTDataContainer, ParallelAnalysisInterface, abc.ABC):
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
        ParallelAnalysisInterface.__init__(self)
        super().__init__(ds, field_parameters)
        self._data_source = data_source
        if data_source is not None:
            if data_source.ds != self.ds:
                raise RuntimeError(
                    "Attempted to construct a DataContainer with a data_source "
                    "from a different Dataset",
                    ds,
                    data_source.ds,
                )
            if data_source._dimensionality < self._dimensionality:
                raise RuntimeError(
                    "Attempted to construct a DataContainer with a data_source "
                    "of lower dimensionality (%u vs %u)"
                    % (data_source._dimensionality, self._dimensionality)
                )
            self.field_parameters.update(data_source.field_parameters)
        self.quantities = DerivedQuantityCollection(self)

    @property
    def selector(self):
        if self._selector is not None:
            return self._selector
        s_module = getattr(self, "_selector_module", yt.geometry.selection_routines)
        sclass = getattr(s_module, f"{self._type_name}_selector", None)
        if sclass is None:
            raise YTDataSelectorNotImplemented(self._type_name)

        if self._data_source is not None:
            self._selector = compose_selector(
                self, self._data_source.selector, sclass(self)
            )
        else:
            self._selector = sclass(self)
        return self._selector

    def chunks(self, fields, chunking_style, **kwargs):
        # This is an iterator that will yield the necessary chunks.
        self.get_data()  # Ensure we have built ourselves
        if fields is None:
            fields = []
        # chunk_ind can be supplied in the keyword arguments.  If it's a
        # scalar, that'll be the only chunk that gets returned; if it's a list,
        # those are the ones that will be.
        chunk_ind = kwargs.pop("chunk_ind", None)
        if chunk_ind is not None:
            chunk_ind = list(always_iterable(chunk_ind))
        for ci, chunk in enumerate(self.index._chunk(self, chunking_style, **kwargs)):
            if chunk_ind is not None and ci not in chunk_ind:
                continue
            with self._chunked_read(chunk):
                self.get_data(fields)
                # NOTE: we yield before releasing the context
                yield self

    def _identify_dependencies(self, fields_to_get, spatial=False):
        inspected = 0
        fields_to_get = fields_to_get[:]
        for field in itertools.cycle(fields_to_get):
            if inspected >= len(fields_to_get):
                break
            inspected += 1
            fi = self.ds._get_field_info(field)
            fd = self.ds.field_dependencies.get(
                field, None
            ) or self.ds.field_dependencies.get(field[1], None)
            # This is long overdue.  Any time we *can't* find a field
            # dependency -- for instance, if the derived field has been added
            # after dataset instantiation -- let's just try to
            # recalculate it.
            if fd is None:
                try:
                    fd = fi.get_dependencies(ds=self.ds)
                    self.ds.field_dependencies[field] = fd
                except Exception:
                    continue
            requested = self._determine_fields(list(set(fd.requested)))
            deps = [d for d in requested if d not in fields_to_get]
            fields_to_get += deps
        return sorted(fields_to_get)

    def get_data(self, fields=None):
        if self._current_chunk is None:
            self.index._identify_base_chunk(self)
        if fields is None:
            return
        nfields = []
        defaultdict(list)
        for field in self._determine_fields(fields):
            # We need to create the field on the raw particle types
            # for particles types (when the field is not directly
            # defined for the derived particle type only)
            finfo = self.ds.field_info[field]

            nfields.append(field)

        fields = nfields
        if len(fields) == 0:
            return
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
            try:
                finfo.check_available(self)
            except NeedsGridType:
                fields_to_generate.append(field)
                continue
            fields_to_get.append(field)
        if len(fields_to_get) == 0 and len(fields_to_generate) == 0:
            return
        elif self._locked:
            raise GenerationInProgress(fields)
        # Track which ones we want in the end
        ofields = set(list(self.field_data.keys()) + fields_to_get + fields_to_generate)
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

        fields_to_generate += gen_fluids
        self._generate_fields(fields_to_generate)
        for field in list(self.field_data.keys()):
            if field not in ofields:
                self.field_data.pop(field)

    def _generate_fields(self, fields_to_generate):
        index = 0

        def dimensions_compare_equal(a, b, /) -> bool:
            if a == b:
                return True
            try:
                if (a == 1 and b.is_dimensionless) or (a.is_dimensionless and b == 1):
                    return True
            except AttributeError:
                return False
            return False

        with self._field_lock():
            # At this point, we assume that any fields that are necessary to
            # *generate* a field are in fact already available to us.  Note
            # that we do not make any assumption about whether or not the
            # fields have a spatial requirement.  This will be checked inside
            # _generate_field, at which point additional dependencies may
            # actually be noted.
            while any(f not in self.field_data for f in fields_to_generate):
                field = fields_to_generate[index % len(fields_to_generate)]
                index += 1
                if field in self.field_data:
                    continue
                fi = self.ds._get_field_info(field)
                try:
                    fd = self._generate_field(field)
                    if hasattr(fd, "units"):
                        fd.units.registry = self.ds.unit_registry
                    if fd is None:
                        raise RuntimeError
                    if fi.units is None:
                        # first time calling a field with units='auto', so we
                        # infer the units from the units of the data we get back
                        # from the field function and use these units for future
                        # field accesses
                        units = getattr(fd, "units", "")
                        if units == "":
                            sunits = ""
                            dimensions = 1
                        else:
                            sunits = str(
                                units.get_base_equivalent(self.ds.unit_system.name)
                            )
                            dimensions = units.dimensions

                        if not dimensions_compare_equal(fi.dimensions, dimensions):
                            raise YTDimensionalityError(fi.dimensions, dimensions)
                        fi.units = sunits
                        fi.dimensions = dimensions
                        self.field_data[field] = self.ds.arr(fd, units)
                    if fi.output_units is None:
                        fi.output_units = fi.units

                    try:
                        fd.convert_to_units(fi.units)
                    except AttributeError:
                        # If the field returns an ndarray, coerce to a
                        # dimensionless YTArray and verify that field is
                        # supposed to be unitless
                        fd = self.ds.arr(fd, "")
                        if fi.units != "":
                            raise YTFieldUnitError(fi, fd.units) from None
                    except UnitConversionError as e:
                        raise YTFieldUnitError(fi, fd.units) from e
                    except UnitParseError as e:
                        raise YTFieldUnitParseError(fi) from e
                    self.field_data[field] = fd
                except GenerationInProgress as gip:
                    for f in gip.fields:
                        if f not in fields_to_generate:
                            fields_to_generate.append(f)

    def __or__(self, other):
        if not isinstance(other, YTSelectionContainer):
            raise YTBooleanObjectError(other)
        if self.ds is not other.ds:
            raise YTBooleanObjectsWrongDataset()
        # Should maybe do something with field parameters here
        from yt.data_objects.selection_objects.boolean_operations import (
            YTBooleanContainer,
        )

        return YTBooleanContainer("OR", self, other, ds=self.ds)

    def __invert__(self):
        # ~obj
        asel = yt.geometry.selection_routines.AlwaysSelector(self.ds)
        from yt.data_objects.selection_objects.boolean_operations import (
            YTBooleanContainer,
        )

        return YTBooleanContainer("NOT", self, asel, ds=self.ds)

    def __xor__(self, other):
        if not isinstance(other, YTSelectionContainer):
            raise YTBooleanObjectError(other)
        if self.ds is not other.ds:
            raise YTBooleanObjectsWrongDataset()
        from yt.data_objects.selection_objects.boolean_operations import (
            YTBooleanContainer,
        )

        return YTBooleanContainer("XOR", self, other, ds=self.ds)

    def __and__(self, other):
        if not isinstance(other, YTSelectionContainer):
            raise YTBooleanObjectError(other)
        if self.ds is not other.ds:
            raise YTBooleanObjectsWrongDataset()
        from yt.data_objects.selection_objects.boolean_operations import (
            YTBooleanContainer,
        )

        return YTBooleanContainer("AND", self, other, ds=self.ds)

    def __add__(self, other):
        return self.__or__(other)

    def __sub__(self, other):
        if not isinstance(other, YTSelectionContainer):
            raise YTBooleanObjectError(other)
        if self.ds is not other.ds:
            raise YTBooleanObjectsWrongDataset()
        from yt.data_objects.selection_objects.boolean_operations import (
            YTBooleanContainer,
        )

        return YTBooleanContainer("NEG", self, other, ds=self.ds)

    @contextmanager
    def _field_lock(self):
        self._locked = True
        yield
        self._locked = False

    @contextmanager
    def _ds_hold(self, new_ds):
        """
        This contextmanager is used to take a data object and preserve its
        attributes but allow the dataset that underlies it to be swapped out.
        This is typically only used internally, and differences in unit systems
        may present interesting possibilities.
        """
        old_ds = self.ds
        old_index = self._index
        self.ds = new_ds
        self._index = new_ds.index
        old_chunk_info = self._chunk_info
        old_chunk = self._current_chunk
        old_size = self.size
        self._chunk_info = None
        self._current_chunk = None
        self.size = None
        self._index._identify_base_chunk(self)
        with self._chunked_read(None):
            yield
        self._index = old_index
        self.ds = old_ds
        self._chunk_info = old_chunk_info
        self._current_chunk = old_chunk
        self.size = old_size

    @contextmanager
    def _chunked_read(self, chunk):
        # There are several items that need to be swapped out
        # field_data, size, shape
        obj_field_data = []
        if hasattr(chunk, "objs"):
            for obj in chunk.objs:
                obj_field_data.append(obj.field_data)
                obj.field_data = YTFieldData()
        old_field_data, self.field_data = self.field_data, YTFieldData()
        old_chunk, self._current_chunk = self._current_chunk, chunk
        old_locked, self._locked = self._locked, False
        yield
        self.field_data = old_field_data
        self._current_chunk = old_chunk
        self._locked = old_locked
        if hasattr(chunk, "objs"):
            for obj in chunk.objs:
                obj.field_data = obj_field_data.pop(0)

    @contextmanager
    def _activate_cache(self):
        cache = self._field_cache or {}
        old_fields = {}
        for field in (f for f in cache if f in self.field_data):
            old_fields[field] = self.field_data[field]
        self.field_data.update(cache)
        yield
        for field in cache:
            self.field_data.pop(field)
            if field in old_fields:
                self.field_data[field] = old_fields.pop(field)
        self._field_cache = None

    def _initialize_cache(self, cache):
        # Wipe out what came before
        self._field_cache = {}
        self._field_cache.update(cache)

    @property
    def icoords(self):
        if self._current_chunk is None:
            self.index._identify_base_chunk(self)
        return self._current_chunk.icoords

    @property
    def fcoords(self):
        if self._current_chunk is None:
            self.index._identify_base_chunk(self)
        return self._current_chunk.fcoords

    @property
    def ires(self):
        if self._current_chunk is None:
            self.index._identify_base_chunk(self)
        return self._current_chunk.ires

    @property
    def fwidth(self):
        if self._current_chunk is None:
            self.index._identify_base_chunk(self)
        return self._current_chunk.fwidth

    @property
    def fcoords_vertex(self):
        if self._current_chunk is None:
            self.index._identify_base_chunk(self)
        return self._current_chunk.fcoords_vertex

    @property
    def max_level(self):
        if self._max_level is None:
            try:
                return self.ds.max_level
            except AttributeError:
                return None
        return self._max_level

    @max_level.setter
    def max_level(self, value):
        if self._selector is not None:
            del self._selector
            self._selector = None
        self._current_chunk = None
        self.size = None
        self.shape = None
        self.field_data.clear()
        self._max_level = value

    @property
    def min_level(self):
        if self._min_level is None:
            try:
                return 0
            except AttributeError:
                return None
        return self._min_level

    @min_level.setter
    def min_level(self, value):
        if self._selector is not None:
            del self._selector
            self._selector = None
        self.field_data.clear()
        self.size = None
        self.shape = None
        self._current_chunk = None
        self._min_level = value


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

    def cut_region(self, field_cuts, field_parameters=None, locals=None):
        """
        Return a YTCutRegion, where the a cell is identified as being inside
        the cut region based on the value of one or more fields.  Note that in
        previous versions of yt the name 'grid' was used to represent the data
        object used to construct the field cut, as of yt 3.0, this has been
        changed to 'obj'.

        Parameters
        ----------
        field_cuts : list of strings
           A list of conditionals that will be evaluated. In the namespace
           available, these conditionals will have access to 'obj' which is a
           data object of unknown shape, and they must generate a boolean array.
           For instance, conditionals = ["obj[('gas', 'temperature')] < 1e3"]
        field_parameters : dictionary
           A dictionary of field parameters to be used when applying the field
           cuts.
        locals : dictionary
            A dictionary of local variables to use when defining the cut region.

        Examples
        --------
        To find the total mass of hot gas with temperature greater than 10^6 K
        in your volume:

        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.cut_region(["obj[('gas', 'temperature')] > 1e6"])
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        if locals is None:
            locals = {}
        cr = self.ds.cut_region(
            self, field_cuts, field_parameters=field_parameters, locals=locals
        )
        return cr

    def _build_operator_cut(self, operation, field, value, units=None):
        """
        Given an operation (>, >=, etc.), a field and a value,
        return the cut_region implementing it.

        This is only meant to be used internally.

        Examples
        --------
        >>> ds._build_operator_cut(">", ("gas", "density"), 1e-24)
        ... # is equivalent to
        ... ds.cut_region(['obj[("gas", "density")] > 1e-24'])
        """
        ftype, fname = self._determine_fields(field)[0]
        if units is None:
            field_cuts = f'obj["{ftype}", "{fname}"] {operation} {value}'
        else:
            field_cuts = (
                f'obj["{ftype}", "{fname}"].in_units("{units}") {operation} {value}'
            )
        return self.cut_region(field_cuts)

    def _build_function_cut(self, function, field, units=None, **kwargs):
        """
        Given a function (np.abs, np.all) and a field,
        return the cut_region implementing it.

        This is only meant to be used internally.

        Examples
        --------
        >>> ds._build_function_cut("np.isnan", ("gas", "density"), locals={"np": np})
        ... # is equivalent to
        ... ds.cut_region(['np.isnan(obj[("gas", "density")])'], locals={"np": np})
        """
        ftype, fname = self._determine_fields(field)[0]
        if units is None:
            field_cuts = f'{function}(obj["{ftype}", "{fname}"])'
        else:
            field_cuts = f'{function}(obj["{ftype}", "{fname}"].in_units("{units}"))'
        return self.cut_region(field_cuts, **kwargs)

    def exclude_above(self, field, value, units=None):
        """
        This function will return a YTCutRegion where all of the regions
        whose field is above a given value are masked.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        value : float
            The minimum value that will not be masked in the output
            YTCutRegion.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the field above the given value masked.

        Examples
        --------
        To find the total mass of hot gas with temperature colder than 10^6 K
        in your volume:

        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.exclude_above(("gas", "temperature"), 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))

        """
        return self._build_operator_cut("<=", field, value, units)

    def include_above(self, field, value, units=None):
        """
        This function will return a YTCutRegion where only the regions
        whose field is above a given value are included.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        value : float
            The minimum value that will not be masked in the output
            YTCutRegion.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the field above the given value masked.

        Examples
        --------
        To find the total mass of hot gas with temperature warmer than 10^6 K
        in your volume:

        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.include_above(("gas", "temperature"), 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """

        return self._build_operator_cut(">", field, value, units)

    def exclude_equal(self, field, value, units=None):
        """
        This function will return a YTCutRegion where all of the regions
        whose field are equal to given value are masked.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        value : float
            The minimum value that will not be masked in the output
            YTCutRegion.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the field equal to the given value masked.

        Examples
        --------
        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.exclude_equal(("gas", "temperature"), 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        return self._build_operator_cut("!=", field, value, units)

    def include_equal(self, field, value, units=None):
        """
        This function will return a YTCutRegion where only the regions
        whose field are equal to given value are included.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        value : float
            The minimum value that will not be masked in the output
            YTCutRegion.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the field equal to the given value included.

        Examples
        --------
        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.include_equal(("gas", "temperature"), 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        return self._build_operator_cut("==", field, value, units)

    def exclude_inside(self, field, min_value, max_value, units=None):
        """
        This function will return a YTCutRegion where all of the regions
        whose field are inside the interval from min_value to max_value.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        min_value : float
            The minimum value inside the interval to be excluded.
        max_value : float
            The maximum value inside the interval to be excluded.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the field inside the given interval excluded.

        Examples
        --------
        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.exclude_inside(("gas", "temperature"), 1e5, 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        ftype, fname = self._determine_fields(field)[0]
        if units is None:
            field_cuts = (
                f'(obj["{ftype}", "{fname}"] <= {min_value}) | '
                f'(obj["{ftype}", "{fname}"] >= {max_value})'
            )
        else:
            field_cuts = (
                f'(obj["{ftype}", "{fname}"].in_units("{units}") <= {min_value}) | '
                f'(obj["{ftype}", "{fname}"].in_units("{units}") >= {max_value})'
            )
        cr = self.cut_region(field_cuts)
        return cr

    def include_inside(self, field, min_value, max_value, units=None):
        """
        This function will return a YTCutRegion where only the regions
        whose field are inside the interval from min_value to max_value are
        included.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        min_value : float
            The minimum value inside the interval to be excluded.
        max_value : float
            The maximum value inside the interval to be excluded.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the field inside the given interval excluded.

        Examples
        --------
        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.include_inside(("gas", "temperature"), 1e5, 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        ftype, fname = self._determine_fields(field)[0]
        if units is None:
            field_cuts = (
                f'(obj["{ftype}", "{fname}"] > {min_value}) & '
                f'(obj["{ftype}", "{fname}"] < {max_value})'
            )
        else:
            field_cuts = (
                f'(obj["{ftype}", "{fname}"].in_units("{units}") > {min_value}) & '
                f'(obj["{ftype}", "{fname}"].in_units("{units}") < {max_value})'
            )
        cr = self.cut_region(field_cuts)
        return cr

    def exclude_outside(self, field, min_value, max_value, units=None):
        """
        This function will return a YTCutRegion where all of the regions
        whose field are outside the interval from min_value to max_value.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        min_value : float
            The minimum value inside the interval to be excluded.
        max_value : float
            The maximum value inside the interval to be excluded.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the field outside the given interval excluded.

        Examples
        --------
        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.exclude_outside(("gas", "temperature"), 1e5, 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        cr = self.exclude_below(field, min_value, units)
        cr = cr.exclude_above(field, max_value, units)
        return cr

    def include_outside(self, field, min_value, max_value, units=None):
        """
        This function will return a YTCutRegion where only the regions
        whose field are outside the interval from min_value to max_value are
        included.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        min_value : float
            The minimum value inside the interval to be excluded.
        max_value : float
            The maximum value inside the interval to be excluded.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the field outside the given interval excluded.

        Examples
        --------
        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.exclude_outside(("gas", "temperature"), 1e5, 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        cr = self.exclude_inside(field, min_value, max_value, units)
        return cr

    def exclude_below(self, field, value, units=None):
        """
        This function will return a YTCutRegion where all of the regions
        whose field is below a given value are masked.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        value : float
            The minimum value that will not be masked in the output
            YTCutRegion.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the field below the given value masked.

        Examples
        --------
        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.exclude_below(("gas", "temperature"), 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        return self._build_operator_cut(">=", field, value, units)

    def exclude_nan(self, field, units=None):
        """
        This function will return a YTCutRegion where all of the regions
        whose field is NaN are masked.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with the NaN entries of the field masked.

        Examples
        --------
        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.exclude_nan(("gas", "temperature"))
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        return self._build_function_cut("~np.isnan", field, units, locals={"np": np})

    def include_below(self, field, value, units=None):
        """
        This function will return a YTCutRegion where only the regions
        whose field is below a given value are included.

        Parameters
        ----------
        field : string
            The field in which the conditional will be applied.
        value : float
            The minimum value that will not be masked in the output
            YTCutRegion.
        units : string or None
            The units of the value threshold. None will use the default units
            given in the field.

        Returns
        -------
        cut_region : YTCutRegion
            The YTCutRegion with only regions with the field below the given
            value included.

        Examples
        --------
        >>> ds = yt.load("RedshiftOutput0005")
        >>> ad = ds.all_data()
        >>> cr = ad.include_below(("gas", "temperature"), 1e5, 1e6)
        >>> print(cr.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun"))
        """
        return self._build_operator_cut("<", field, value, units)

    def extract_isocontours(
        self, field, value, filename=None, rescale=False, sample_values=None
    ):
        r"""This identifies isocontours on a cell-by-cell basis, with no
        consideration of global connectedness, and returns the vertices of the
        Triangles in that isocontour.

        This function simply returns the vertices of all the triangles
        calculated by the `marching cubes
        <https://en.wikipedia.org/wiki/Marching_cubes>`_ algorithm; for more
        complex operations, such as identifying connected sets of cells above a
        given threshold, see the extract_connected_sets function.  This is more
        useful for calculating, for instance, total isocontour area, or
        visualizing in an external program (such as `MeshLab
        <http://www.meshlab.net>`_.)

        Parameters
        ----------
        field : string
            Any field that can be obtained in a data object.  This is the field
            which will be isocontoured.
        value : float
            The value at which the isocontour should be calculated.
        filename : string, optional
            If supplied, this file will be filled with the vertices in .obj
            format.  Suitable for loading into meshlab.
        rescale : bool, optional
            If true, the vertices will be rescaled within their min/max.
        sample_values : string, optional
            Any field whose value should be extracted at the center of each
            triangle.

        Returns
        -------
        verts : array of floats
            The array of vertices, x,y,z.  Taken in threes, these are the
            triangle vertices.
        samples : array of floats
            If `sample_values` is specified, this will be returned and will
            contain the values of the field specified at the center of each
            triangle.

        Examples
        --------
        This will create a data object, find a nice value in the center, and
        output the vertices to "triangles.obj" after rescaling them.

        >>> dd = ds.all_data()
        >>> rho = dd.quantities["WeightedAverageQuantity"](
        ...     ("gas", "density"), weight=("gas", "cell_mass")
        ... )
        >>> verts = dd.extract_isocontours(
        ...     ("gas", "density"), rho, "triangles.obj", True
        ... )
        """
        from yt.data_objects.static_output import ParticleDataset
        from yt.frontends.stream.data_structures import StreamParticlesDataset

        verts = []
        samples = []
        if isinstance(self.ds, (ParticleDataset, StreamParticlesDataset)):
            raise NotImplementedError
        for block, mask in self.blocks:
            my_verts = self._extract_isocontours_from_grid(
                block, mask, field, value, sample_values
            )
            if sample_values is not None:
                my_verts, svals = my_verts
                samples.append(svals)
            verts.append(my_verts)
        verts = np.concatenate(verts).transpose()
        verts = self.comm.par_combine_object(verts, op="cat", datatype="array")
        verts = verts.transpose()
        if sample_values is not None:
            samples = np.concatenate(samples)
            samples = self.comm.par_combine_object(samples, op="cat", datatype="array")
        if rescale:
            mi = np.min(verts, axis=0)
            ma = np.max(verts, axis=0)
            verts = (verts - mi) / (ma - mi).max()
        if filename is not None and self.comm.rank == 0:
            if hasattr(filename, "write"):
                f = filename
            else:
                f = open(filename, "w")
            for v1 in verts:
                f.write(f"v {v1[0]:0.16e} {v1[1]:0.16e} {v1[2]:0.16e}\n")
            for i in range(len(verts) // 3):
                f.write(f"f {i * 3 + 1} {i * 3 + 2} {i * 3 + 3}\n")
            if not hasattr(filename, "write"):
                f.close()
        if sample_values is not None:
            return verts, samples
        return verts

    def _extract_isocontours_from_grid(
        self, grid, mask, field, value, sample_values=None
    ):
        vc_fields = [field]
        if sample_values is not None:
            vc_fields.append(sample_values)

        vc_data = grid.get_vertex_centered_data(vc_fields, no_ghost=False)
        try:
            svals = vc_data[sample_values]
        except KeyError:
            svals = None

        my_verts = march_cubes_grid(
            value, vc_data[field], mask, grid.LeftEdge, grid.dds, svals
        )
        return my_verts

    def calculate_isocontour_flux(
        self, field, value, field_x, field_y, field_z, fluxing_field=None
    ):
        r"""This identifies isocontours on a cell-by-cell basis, with no
        consideration of global connectedness, and calculates the flux over
        those contours.

        This function will conduct `marching cubes
        <https://en.wikipedia.org/wiki/Marching_cubes>`_ on all the cells in a
        given data container (grid-by-grid), and then for each identified
        triangular segment of an isocontour in a given cell, calculate the
        gradient (i.e., normal) in the isocontoured field, interpolate the local
        value of the "fluxing" field, the area of the triangle, and then return:

        area * local_flux_value * (n dot v)

        Where area, local_value, and the vector v are interpolated at the barycenter
        (weighted by the vertex values) of the triangle.  Note that this
        specifically allows for the field fluxing across the surface to be
        *different* from the field being contoured.  If the fluxing_field is
        not specified, it is assumed to be 1.0 everywhere, and the raw flux
        with no local-weighting is returned.

        Additionally, the returned flux is defined as flux *into* the surface,
        not flux *out of* the surface.

        Parameters
        ----------
        field : string
            Any field that can be obtained in a data object.  This is the field
            which will be isocontoured and used as the "local_value" in the
            flux equation.
        value : float
            The value at which the isocontour should be calculated.
        field_x : string
            The x-component field
        field_y : string
            The y-component field
        field_z : string
            The z-component field
        fluxing_field : string, optional
            The field whose passage over the surface is of interest.  If not
            specified, assumed to be 1.0 everywhere.

        Returns
        -------
        flux : float
            The summed flux.  Note that it is not currently scaled; this is
            simply the code-unit area times the fields.

        Examples
        --------
        This will create a data object, find a nice value in the center, and
        calculate the metal flux over it.

        >>> dd = ds.all_data()
        >>> rho = dd.quantities["WeightedAverageQuantity"](
        ...     ("gas", "density"), weight=("gas", "cell_mass")
        ... )
        >>> flux = dd.calculate_isocontour_flux(
        ...     ("gas", "density"),
        ...     rho,
        ...     ("gas", "velocity_x"),
        ...     ("gas", "velocity_y"),
        ...     ("gas", "velocity_z"),
        ...     ("gas", "metallicity"),
        ... )
        """
        flux = 0.0
        for block, mask in self.blocks:
            flux += self._calculate_flux_in_grid(
                block, mask, field, value, field_x, field_y, field_z, fluxing_field
            )
        flux = self.comm.mpi_allreduce(flux, op="sum")
        return flux

    def _calculate_flux_in_grid(
        self, grid, mask, field, value, field_x, field_y, field_z, fluxing_field=None
    ):
        vc_fields = [field, field_x, field_y, field_z]
        if fluxing_field is not None:
            vc_fields.append(fluxing_field)

        vc_data = grid.get_vertex_centered_data(vc_fields)

        if fluxing_field is None:
            ff = np.ones_like(vc_data[field], dtype="float64")
        else:
            ff = vc_data[fluxing_field]

        return march_cubes_grid_flux(
            value,
            vc_data[field],
            vc_data[field_x],
            vc_data[field_y],
            vc_data[field_z],
            ff,
            mask,
            grid.LeftEdge,
            grid.dds,
        )

    def extract_connected_sets(
        self, field, num_levels, min_val, max_val, log_space=True, cumulative=True
    ):
        """
        This function will create a set of contour objects, defined
        by having connected cell structures, which can then be
        studied and used to 'paint' their source grids, thus enabling
        them to be plotted.

        Note that this function *can* return a connected set object that has no
        member values.
        """
        if log_space:
            cons = np.logspace(np.log10(min_val), np.log10(max_val), num_levels + 1)
        else:
            cons = np.linspace(min_val, max_val, num_levels + 1)
        contours = {}
        for level in range(num_levels):
            contours[level] = {}
            if cumulative:
                mv = max_val
            else:
                mv = cons[level + 1]
            from yt.data_objects.level_sets.api import identify_contours
            from yt.data_objects.level_sets.clump_handling import add_contour_field

            nj, cids = identify_contours(self, field, cons[level], mv)
            unique_contours = set()
            for sl_list in cids.values():
                for _sl, ff in sl_list:
                    unique_contours.update(np.unique(ff))
            contour_key = uuid.uuid4().hex
            # In case we're a cut region already...
            base_object = getattr(self, "base_object", self)
            add_contour_field(base_object.ds, contour_key)
            for cid in sorted(unique_contours):
                if cid == -1:
                    continue
                contours[level][cid] = base_object.cut_region(
                    [f"obj['contours_{contour_key}'] == {cid}"],
                    {f"contour_slices_{contour_key}": cids},
                )
        return cons, contours

    def _get_bbox(self):
        """
        Return the bounding box for this data container.
        This generic version will return the bounds of the entire domain.
        """
        return self.ds.domain_left_edge, self.ds.domain_right_edge

    def get_bbox(self) -> tuple[unyt_array, unyt_array]:
        """
        Return the bounding box for this data container.
        """
        le, re = self._get_bbox()
        le.convert_to_units("code_length")
        re.convert_to_units("code_length")
        return le, re


    def volume(self):
        """
        Return the volume of the data container.
        This is found by adding up the volume of the cells with centers
        in the container, rather than using the geometric shape of
        the container, so this may vary very slightly
        from what might be expected from the geometric volume.
        """
        return self.quantities.total_quantity(("index", "cell_volume"))


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
        if not isinstance(left_edge, YTArray):
            self.left_edge = self.ds.arr(left_edge, "code_length", dtype="float64")
        else:
            # need to assign this dataset's unit registry to the YTArray
            self.left_edge = self.ds.arr(left_edge.copy(), dtype="float64")
        if not isinstance(right_edge, YTArray):
            self.right_edge = self.ds.arr(right_edge, "code_length", dtype="float64")
        else:
            # need to assign this dataset's unit registry to the YTArray
            self.right_edge = self.ds.arr(right_edge.copy(), dtype="float64")

    def _get_bbox(self):
        """
        Return the minimum bounding box for the region.
        """
        return self.left_edge.copy(), self.right_edge.copy()


class MutableAttribute:
    """A descriptor for mutable data"""

    def __init__(self, display_array=False):
        self.data = weakref.WeakKeyDictionary()
        # We can assume that ipywidgets will not be *added* to the system
        # during the course of execution, and if it is, we will not wrap the
        # array.
        if display_array and find_spec("ipywidgets") is not None:
            self.display_array = True
        else:
            self.display_array = False

    def __get__(self, instance, owner):
        if not instance:
            return None
        ret = self.data.get(instance, None)
        try:
            ret = ret.copy()
        except AttributeError:
            pass
        if self.display_array:
            try:
                ret._ipython_display_ = functools.partial(_wrap_display_ytarray, ret)
            # This will error out if the items have yet to be turned into
            # YTArrays, in which case we just let it go.
            except AttributeError:
                pass
        return ret

    def __set__(self, instance, value):
        self.data[instance] = value


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
    particle_unions: Optional[dict[ParticleType, ParticleUnion]] = None
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

    # these are set in self._parse_parameter_file()
    domain_left_edge = MutableAttribute(True)
    domain_right_edge = MutableAttribute(True)
    domain_dimensions = MutableAttribute(True)
    # the point in index space "domain_left_edge" doesn't necessarily
    # map to (0, 0, 0)
    domain_offset = np.zeros(3, dtype="int64")
    _periodicity = MutableAttribute()
    _force_periodicity = False

    # these are set in self._set_derived_attrs()
    domain_width = MutableAttribute(True)
    domain_center = MutableAttribute(True)

    def __init__(
        self,
        filename: str,
        dataset_type: Optional[str] = None,
        units_override: Optional[dict[str, str]] = None,
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
        ] = "cgs",
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
        if self._instantiated != 0:
            return
        self.dataset_type = dataset_type
        self.conversion_factors = {}
        self.parameters: dict[str, Any] = {}
        self.region_expression = self.r = RegionExpression(self)
        self.particle_unions = self.particle_unions or {}
        self.field_units = self.field_units or {}
        self._determined_fields = {}
        self.units_override = units_override
        self.default_species_fields = default_species_fields

        self._input_filename: str = os.fspath(filename)

        # to get the timing right, do this before the heavy lifting
        self._instantiated = time.time()

        self.no_cgs_equiv_length = False

        if unit_system == "code":
            # create a fake MKS unit system which we will override later to
            # avoid chicken/egg issue of the unit registry needing a unit system
            # but code units need a unit registry to define the code units on
            used_unit_system = "mks"
        else:
            used_unit_system = unit_system

        self._create_unit_registry(used_unit_system)

        self._parse_parameter_file()
        self.set_units()
        self.setup_cosmology()
        self._assign_unit_system(unit_system)
        self._setup_coordinate_handler(axis_order)
        self._set_derived_attrs()
        # Because we need an instantiated class to check the ds's existence in
        # the cache, we move that check to here from __new__.  This avoids
        # double-instantiation.
        # PR 3124: _set_derived_attrs() can change the hash, check store here
        if _ds_store is None:
            raise RuntimeError(
                "Something went wrong during yt's initialization: "
                "dataset cache isn't properly initialized"
            )
        try:
            _ds_store.check_ds(self)
        except NoParameterShelf:
            pass
        self._setup_classes()

    @property
    def filename(self):
        if self._input_filename.startswith("http"):
            return self._input_filename
        else:
            return os.path.abspath(os.path.expanduser(self._input_filename))

    @property
    def basename(self):
        return os.path.basename(self.filename)

    @property
    def directory(self):
        return os.path.dirname(self.filename)

    @property
    def periodicity(self):
        return self._periodicity

    @abc.abstractmethod
    def _parse_parameter_file(self):
        # set up various attributes from self.parameter_filename
        # for a full description of what is required here see
        # yt.frontends._skeleton.SkeletonDataset
        pass

    @abc.abstractmethod
    def _set_code_unit_attributes(self):
        # set up code-units to physical units normalization factors
        # for a full description of what is required here see
        # yt.frontends._skeleton.SkeletonDataset
        pass

    def _set_derived_attrs(self):
        if self.domain_left_edge is None or self.domain_right_edge is None:
            self.domain_center = np.zeros(3)
            self.domain_width = np.zeros(3)
        else:
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
        s = f"{self.basename};{self.current_time};{self.unique_identifier}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def __getitem__(self, key):
        """Returns units, parameters, or conversion_factors in that order."""
        return self.parameters[key]

    def __iter__(self):
        yield from self.parameters

    def has_key(self, key):
        """
        Checks units, parameters, and conversion factors. Returns a boolean.

        """
        return key in self.parameters

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

        self.field_info.setup_extra_union_fields()
        self.field_info.load_all_plugins(self.default_fluid_type)
        deps, unloaded = self.field_info.check_derived_fields()
        self.field_dependencies.update(deps)
        self.fields = FieldTypeContainer(self)
        self.index.field_list = sorted(self.field_list)
        # Now that we've detected the fields, set this flag so that
        # deprecated fields will be logged if they are used
        self.fields_detected = True

    def set_field_label_format(self, format_property, value):
        """
        Set format properties for how fields will be written
        out. Accepts

        format_property : string indicating what property to set
        value: the value to set for that format_property
        """
        available_formats = {"ionization_label": ("plus_minus", "roman_numeral")}
        if format_property in available_formats:
            if value in available_formats[format_property]:
                setattr(self, f"_{format_property}_format", value)
            else:
                raise ValueError(
                    "{} not an acceptable value for format_property "
                    "{}. Choices are {}.".format(
                        value, format_property, available_formats[format_property]
                    )
                )
        else:
            raise ValueError(
                f"{format_property} not a recognized format_property. Available "
                f"properties are: {list(available_formats.keys())}"
            )

    def _setup_coordinate_handler(self, axis_order: Optional[AxisOrder]) -> None:
        self.coordinates = CartesianCoordinateHandler(self, ordering=axis_order)

    def _get_field_info(
        self,
        field: Union[FieldKey, ImplicitFieldKey, DerivedField],
        /,
    ) -> DerivedField:
        field_info, candidates = self._get_field_info_helper(field)

        if field_info.name[1] in ("px", "py", "pz", "pdx", "pdy", "pdz"):
            # escape early as a bandaid solution to
            # https://github.com/yt-project/yt/issues/3381
            return field_info

        return field_info

    def _get_field_info_helper(
        self,
        field: Union[FieldKey, ImplicitFieldKey, DerivedField],
        /,
    ) -> tuple[DerivedField, list[FieldKey]]:
        self.index

        ftype: str
        fname: str
        if isinstance(field, str):
            ftype, fname = "unknown", field
        elif isinstance(field, tuple) and len(field) == 2:
            ftype, fname = field
        elif isinstance(field, DerivedField):
            ftype, fname = field.name
        else:
            raise YTFieldNotParseable(field)

        if (ftype, fname) in self.field_info:
            return self.field_info[ftype, fname], []

        raise YTFieldNotFound(field, ds=self)

    def _setup_classes(self):
        # Called by subclass
        self.object_types = []
        self.objects = []
        self.plots = []
        for name, cls in sorted(data_object_registry.items()):
            if name in self._index_class._unsupported_objects:
                setattr(self, name, _unsupported_object(self, name))
                continue
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

    def _find_extremum(self, field, ext, source=None, to_array=True):
        """
        Find the extremum value of a field in a data object (source) and its position.

        Parameters
        ----------
        field : str or tuple(str, str)
        ext : str
            'min' or 'max', select an extremum
        source : a Yt data object
        to_array : bool
            select the return type.

        Returns
        -------
        val, coords

        val: unyt.unyt_quantity
            extremum value detected

        coords: unyt.unyt_array or list(unyt.unyt_quantity)
            Conversion to a single unyt_array object is only possible for coordinate
            systems with homogeneous dimensions across axes (i.e. cartesian).
        """
        ext = ext.lower()
        if source is None:
            source = self.all_data()
        method = {
            "min": source.quantities.min_location,
            "max": source.quantities.max_location,
        }[ext]
        val, x1, x2, x3 = method(field)
        coords = [x1, x2, x3]
        if to_array:
            # force conversion to length
            alt_coords = []
            for x in coords:
                alt_coords.append(
                    self.quan(x.v, "code_length")
                    if x.units.is_dimensionless
                    else x.to("code_length")
                )
            coords = self.arr(alt_coords, dtype="float64").to("code_length")
        return val, coords

    def find_max(self, field, source=None, to_array=True):
        """
        Returns (value, location) of the maximum of a given field.

        This is a wrapper around _find_extremum
        """
        return self._find_extremum(field, "max", source=source, to_array=to_array)

    def find_min(self, field, source=None, to_array=True):
        """
        Returns (value, location) for the minimum of a given field.

        This is a wrapper around _find_extremum
        """
        return self._find_extremum(field, "min", source=source, to_array=to_array)

    def find_field_values_at_point(self, fields, coords):
        """
        Returns the values [field1, field2,...] of the fields at the given
        coordinates. Returns a list of field values in the same order as
        the input *fields*.
        """
        point = self.point(coords)
        ret = [point[f] for f in iter_fields(fields)]
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def find_field_values_at_points(self, fields, coords):
        """
        Returns the values [field1, field2,...] of the fields at the given
        [(x1, y1, z2), (x2, y2, z2),...] points.  Returns a list of field
        values in the same order as the input *fields*.

        """
        # If an optimized version exists on the Index object we'll use that
        try:
            return self.index._find_field_values_at_points(fields, coords)
        except AttributeError:
            pass

        fields = list(iter_fields(fields))
        out = []

        # This may be slow because it creates a data object for each point
        for field_index, field in enumerate(fields):
            funit = self._get_field_info(field).units
            out.append(self.arr(np.empty((len(coords),)), funit))
            for coord_index, coord in enumerate(coords):
                out[field_index][coord_index] = self.point(coord)[field]
        if len(fields) == 1:
            return out[0]
        else:
            return out

    # Now all the object related stuff
    def all_data(self, find_max=False, **kwargs):
        """
        all_data is a wrapper to the Region object for creating a region
        which covers the entire simulation domain.
        """
        self.index
        if find_max:
            c = self.find_max("density")[1]
        else:
            c = (self.domain_right_edge + self.domain_left_edge) / 2.0
        return self.region(c, self.domain_left_edge, self.domain_right_edge, **kwargs)

    def box(self, left_edge, right_edge, **kwargs):
        """
        box is a wrapper to the Region object for creating a region
        without having to specify a *center* value.  It assumes the center
        is the midpoint between the left_edge and right_edge.

        Keyword arguments are passed to the initializer of the YTRegion object
        (e.g. ds.region).
        """
        # we handle units in the region data object
        # but need to check if left_edge or right_edge is a
        # list or other non-array iterable before calculating
        # the center
        if isinstance(left_edge[0], YTQuantity):
            left_edge = YTArray(left_edge)
            right_edge = YTArray(right_edge)

        left_edge = np.asanyarray(left_edge, dtype="float64")
        right_edge = np.asanyarray(right_edge, dtype="float64")
        c = (left_edge + right_edge) / 2.0
        return self.region(c, left_edge, right_edge, **kwargs)

    def _setup_particle_type(self, ptype):
        orig = set(self.field_info.items())
        self.field_info.setup_particle_fields(ptype)
        return [n for n, v in set(self.field_info.items()).difference(orig)]

    @property
    def ires_factor(self):
        o2 = np.log2(self.refine_by)
        if o2 != int(o2):
            raise RuntimeError
        # In the case that refine_by is 1 or 0 or something, we just
        # want to make it a non-operative number, so we set to 1.
        return max(1, int(o2))

    def relative_refinement(self, l0, l1):
        return self.refine_by ** (l1 - l0)

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
        mag_dims: Optional[set[Symbol]]
        if mag_unit is not None:
            mag_dims = mag_unit.units.dimensions.free_symbols
        else:
            mag_dims = None
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
        # Now we get to the tricky part. If we have an MKS-like system but
        # we asked for a conversion to something CGS-like, or vice-versa,
        # we have to convert the magnetic field
        if mag_dims is not None:
            self.magnetic_unit: unyt_quantity
            if mks_system and current_mks not in mag_dims:
                self.magnetic_unit = self.quan(
                    self.magnetic_unit.to_value("gauss") * 1.0e-4, "T"
                )
                # The following modification ensures that we get the conversion to
                # mks correct
                self.unit_registry.modify(
                    "code_magnetic", self.magnetic_unit.value * 1.0e3 * 0.1**-0.5
                )
            elif not mks_system and current_mks in mag_dims:
                self.magnetic_unit = self.quan(
                    self.magnetic_unit.to_value("T") * 1.0e4, "gauss"
                )
                # The following modification ensures that we get the conversion to
                # cgs correct
                self.unit_registry.modify(
                    "code_magnetic", self.magnetic_unit.value * 1.0e-4
                )
        current_mks_unit = "A" if mks_system else None
        us = create_code_unit_system(
            self.unit_registry, current_mks_unit=current_mks_unit
        )
        if unit_system != "code":
            us = unit_system_registry[str(unit_system).lower()]

        self._unit_system_name: str = unit_system

        self.unit_system: UnitSystem = us
        self.unit_registry.unit_system = self.unit_system

    @property
    def _uses_code_length_unit(self) -> bool:
        return self._unit_system_name == "code" or self.no_cgs_equiv_length

    @property
    def _uses_code_time_unit(self) -> bool:
        return self._unit_system_name == "code"

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

        if getattr(self, "cosmological_simulation", False):
            # this dataset is cosmological, so add cosmological units.
            self.unit_registry.modify("h", self.hubble_constant)
            if getattr(self, "current_redshift", None) is not None:
                # Comoving lengths
                for my_unit in ["m", "pc", "AU", "au"]:
                    new_unit = f"{my_unit}cm"
                    my_u = Unit(my_unit, registry=self.unit_registry)
                    self.unit_registry.add(
                        new_unit,
                        my_u.base_value / (1 + self.current_redshift),
                        dimensions.length,
                        "\\rm{%s}/(1+z)" % my_unit,
                        prefixable=True,
                    )
                self.unit_registry.modify("a", 1 / (1 + self.current_redshift))

        self.set_code_units()

    def setup_cosmology(self):
        """
        If this dataset is cosmological, add a cosmology object.
        """
        if not getattr(self, "cosmological_simulation", False):
            return

        # Set dynamical dark energy parameters
        use_dark_factor = getattr(self, "use_dark_factor", False)
        w_0 = getattr(self, "w_0", -1.0)
        w_a = getattr(self, "w_a", 0.0)

        # many frontends do not set this
        setdefaultattr(self, "omega_radiation", 0.0)

        self.cosmology = Cosmology(
            hubble_constant=self.hubble_constant,
            omega_matter=self.omega_matter,
            omega_lambda=self.omega_lambda,
            omega_radiation=self.omega_radiation,
            use_dark_factor=use_dark_factor,
            w_0=w_0,
            w_a=w_a,
        )

        if not hasattr(self, "current_time"):
            self.current_time = self.cosmology.t_from_z(self.current_redshift)

        if getattr(self, "current_redshift", None) is not None:
            self.critical_density = self.cosmology.critical_density(
                self.current_redshift
            )
            self.scale_factor = 1.0 / (1.0 + self.current_redshift)

    def get_unit_from_registry(self, unit_str):
        """
        Creates a unit object matching the string expression, using this
        dataset's unit registry.

        Parameters
        ----------
        unit_str : str
            string that we can parse for a sympy Expr.

        """
        new_unit = Unit(unit_str, registry=self.unit_registry)
        return new_unit

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
        if hasattr(self, "magnetic_unit"):
            if self.magnetic_unit.units.dimensions == dimensions.magnetic_field_cgs:
                # We have to cast this explicitly to MKS base units, otherwise
                # unyt will convert it automatically to Tesla
                value = self.magnetic_unit.to_value("sqrt(kg)/(sqrt(m)*s)")
                dims = dimensions.magnetic_field_cgs
            else:
                value = self.magnetic_unit.to_value("T")
                dims = dimensions.magnetic_field_mks
        else:
            # Fallback to gauss if no magnetic unit is specified
            # 1 gauss = 1 sqrt(g)/(sqrt(cm)*s) = 0.1**0.5 sqrt(kg)/(sqrt(m)*s)
            value = 0.1**0.5
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

    def add_field(
        self, name, function, sampling_type, *, force_override=False, **kwargs
    ):
        """
        Dataset-specific call to add_field

        Add a new field, along with supplemental metadata, to the list of
        available fields.  This respects a number of arguments, all of which
        are passed on to the constructor for
        :class:`~yt.data_objects.api.DerivedField`.

        Parameters
        ----------

        name : str
           is the name of the field.
        function : callable
           A function handle that defines the field.  Should accept
           arguments (field, data)
        sampling_type: str
           "cell" or "particle" or "local"
        force_override: bool
           If False (default), an error will be raised if a field of the same name already exists.
        units : str
           A plain text string encoding the unit.  Powers must be in
           python syntax (** instead of ^).
        take_log : bool
           Describes whether the field should be logged
        validators : list
           A list of :class:`FieldValidator` objects
        vector_field : bool
           Describes the dimensionality of the field.  Currently unused.
        display_name : str
           A name used in the plots
        force_override : bool
           Whether to override an existing derived field. Does not work with
           on-disk fields.

        """
        validate_field_function(function)
        self.index
        if force_override and name in self.index.field_list:
            raise RuntimeError(
                "force_override is only meant to be used with "
                "derived fields, not on-disk fields."
            )

        self.field_info.add_field(
            name, function, sampling_type, force_override=force_override, **kwargs
        )
        self.field_info._show_field_errors.append(name)
        deps, _ = self.field_info.check_derived_fields([name])
        self.field_dependencies.update(deps)


    def define_unit(self, symbol, value, tex_repr=None, offset=None, prefixable=False):
        """
        Define a new unit and add it to the dataset's unit registry.

        Parameters
        ----------
        symbol : string
            The symbol for the new unit.
        value : tuple or ~yt.units.yt_array.YTQuantity
            The definition of the new unit in terms of some other units. For example,
            one would define a new "mph" unit with (1.0, "mile/hr")
        tex_repr : string, optional
            The LaTeX representation of the new unit. If one is not supplied, it will
            be generated automatically based on the symbol string.
        offset : float, optional
            The default offset for the unit. If not set, an offset of 0 is assumed.
        prefixable : bool, optional
            Whether or not the new unit can use SI prefixes. Default: False

        Examples
        --------
        >>> ds.define_unit("mph", (1.0, "mile/hr"))
        >>> two_weeks = YTQuantity(14.0, "days")
        >>> ds.define_unit("fortnight", two_weeks)
        """
        define_unit(
            symbol,
            value,
            tex_repr=tex_repr,
            offset=offset,
            prefixable=prefixable,
            registry=self.unit_registry,
        )

    def _is_within_domain(self, point) -> bool:
        assert len(point) == len(self.domain_left_edge)
        assert point.units.dimensions == un.dimensions.length
        for i, x in enumerate(point):
            if self.periodicity[i]:
                continue
            if x < self.domain_left_edge[i]:
                return False
            if x > self.domain_right_edge[i]:
                return False
        return True


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
        if self.ds is None:
            return
        index_fields = {f for _, f in self if _ == "index"}
        for ftype in self.ds.fluid_types:
            if ftype in ("index", "deposit"):
                continue
            for f in index_fields:
                if (ftype, f) in self:
                    continue
                self.alias((ftype, f), ("index", f))

    def setup_particle_fields(self, ptype, ftype="gas", num_neighbors=64):
        skip_output_units = ("code_length",)
        for f, (units, aliases, dn) in sorted(self.known_particle_fields):
            units = self.ds.field_units.get((ptype, f), units)
            output_units = units
            if (
                f in aliases or ptype not in self.ds.particle_types_raw
            ) and units not in skip_output_units:
                u = Unit(units, registry=self.ds.unit_registry)
                if u.dimensions is not dimensionless:
                    output_units = str(self.ds.unit_system[u.dimensions])
            if (ptype, f) not in self.field_list:
                continue
            self.add_output_field(
                (ptype, f),
                sampling_type="particle",
                units=units,
                display_name=dn,
                output_units=output_units,
            )
            for alias in aliases:
                self.alias((ptype, alias), (ptype, f), units=output_units)

        # We'll either have particle_position or particle_position_[xyz]
        if (ptype, "particle_position") in self.field_list or (
            ptype,
            "particle_position",
        ) in self.field_aliases:
            particle_scalar_functions(
                ptype, "particle_position", "particle_velocity", self
            )
        else:
            # We need to check to make sure that there's a "known field" that
            # overlaps with one of the vector fields.  For instance, if we are
            # in the Stream frontend, and we have a set of scalar position
            # fields, they will overlap with -- and be overridden by -- the
            # "known" vector field that the frontend creates.  So the easiest
            # thing to do is to simply remove the on-disk field (which doesn't
            # exist) and replace it with a derived field.
            if (ptype, "particle_position") in self and self[
                ptype, "particle_position"
            ]._function == NullFunc:
                self.pop((ptype, "particle_position"))
            particle_vector_functions(
                ptype,
                [f"particle_position_{ax}" for ax in "xyz"],
                [f"particle_velocity_{ax}" for ax in "xyz"],
                self,
            )
        particle_deposition_functions(ptype, "particle_position", "particle_mass", self)
        standard_particle_fields(self, ptype)
        # Now we check for any leftover particle fields
        for field in sorted(self.field_list):
            if field in self:
                continue
            if not isinstance(field, tuple):
                raise RuntimeError
            if field[0] not in self.ds.particle_types:
                continue
            units = self.ds.field_units.get(field, None)
            if units is None:
                try:
                    units = ytcfg.get("fields", *field, "units")
                except KeyError:
                    units = ""
            self.add_output_field(
                field,
                sampling_type="particle",
                units=units,
            )
        self.setup_smoothed_fields(ptype, num_neighbors=num_neighbors, ftype=ftype)

    def setup_extra_union_fields(self, ptype="all"):
        if ptype != "all":
            raise RuntimeError(
                "setup_extra_union_fields is currently"
                + 'only enabled for particle type "all".'
            )
        for units, field in self.extra_union_fields:
            add_union_field(self, ptype, field, units)

    def setup_smoothed_fields(self, ptype, num_neighbors=64, ftype="gas"):
        # We can in principle compute this, but it is not yet implemented.
        if (ptype, "density") not in self or not hasattr(self.ds, "_sph_ptypes"):
            return
        new_aliases = []
        for ptype2, alias_name in list(self):
            if ptype2 != ptype:
                continue
            if alias_name not in sph_whitelist_fields:
                if alias_name.startswith("particle_"):
                    pass
                else:
                    continue
            uni_alias_name = alias_name
            if "particle_position_" in alias_name:
                uni_alias_name = alias_name.replace("particle_position_", "")
            elif "particle_" in alias_name:
                uni_alias_name = alias_name.replace("particle_", "")
            new_aliases.append(
                (
                    (ftype, uni_alias_name),
                    (ptype, alias_name),
                )
            )
            if "particle_position_" in alias_name:
                new_aliases.append(
                    (
                        (ftype, alias_name),
                        (ptype, alias_name),
                    )
                )
            new_aliases.append(
                (
                    (ptype, uni_alias_name),
                    (ptype, alias_name),
                )
            )
            for alias, source in new_aliases:
                self.alias(alias, source)
        self.alias((ftype, "particle_position"), (ptype, "particle_position"))
        self.alias((ftype, "particle_mass"), (ptype, "particle_mass"))

    # Collect the names for all aliases if geometry is curvilinear
    def get_aliases_gallery(self) -> list[FieldName]:
        aliases_gallery: list[FieldName] = []
        known_other_fields = dict(self.known_other_fields)

        if self.ds is None:
            return aliases_gallery
        return aliases_gallery

    def setup_fluid_aliases(self, ftype: FieldType = "gas") -> None:
        known_other_fields = dict(self.known_other_fields)

        # For non-Cartesian geometry, convert alias of vector fields to
        # curvilinear coordinates
        aliases_gallery = self.get_aliases_gallery()

        for field in sorted(self.field_list):
            if not isinstance(field, tuple) or len(field) != 2:
                raise RuntimeError
            args = known_other_fields.get(field[1], None)
            if args is not None:
                units, aliases, display_name = args
            else:
                try:
                    node = ytcfg.get("fields", *field).as_dict()
                except KeyError:
                    node = {}

                units = node.get("units", "")
                aliases = node.get("aliases", [])
                display_name = node.get("display_name", None)

            # We allow field_units to override this.  First we check if the
            # field *name* is in there, then the field *tuple*.
            units = self.ds.field_units.get(field[1], units)
            units = self.ds.field_units.get(field, units)
            self.add_output_field(
                field, sampling_type="cell", units=units, display_name=display_name
            )
            axis_names = self.ds.coordinates.axis_order
            geometry: Geometry = self.ds.geometry
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
        if ftype is None:
            return
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
        if name[1] == "density":
            if name in self:
                # this should not happen, but it does
                # it'd be best to raise an error here but
                # it may take a while to cleanup internal issues
                return
        kwargs.setdefault("ds", self.ds)
        self[name] = DerivedField(name, sampling_type, NullFunc, **kwargs)

    def alias(
        self,
        alias_name: FieldKey,
        original_name: FieldKey,
        units: Optional[str] = None,
        deprecate: Optional[tuple[str, Optional[str]]] = None,
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
        if original_name not in self:
            return
        if units is None:
            # We default to CGS here, but in principle, this can be pluggable
            # as well.

            # self[original_name].units may be set to `None` at this point
            # to signal that units should be autoset later
            oru = self[original_name].units
            if oru is None:
                units = None
            else:
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


    def has_key(self, key):
        # This gets used a lot
        if key in self:
            return True
        if self.fallback is None:
            return False
        return key in self.fallback

    def __missing__(self, key):
        if self.fallback is None:
            raise KeyError(f"No field named {key}")
        return self.fallback[key]

    @classmethod
    def create_with_fallback(cls, fallback, name=""):
        obj = cls()
        obj.fallback = fallback
        obj.name = name
        return obj

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if self.fallback is None:
            return False
        return key in self.fallback

    def __iter__(self):
        yield from super().__iter__()
        if self.fallback is not None:
            yield from self.fallback

    def keys(self):
        keys = super().keys()
        if self.fallback:
            keys += list(self.fallback.keys())
        return keys

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
            missing = not all(f in self.field_list for f in fd.requested)
            if missing:
                self.pop(field)
                unavailable.append(field)
                continue
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
                try:
                    ftype, fname = field
                    if "vertex" in fname:
                        continue
                except ValueError:
                    # in very rare cases, there can a field represented by a single
                    # string, like "emissivity"
                    # this try block _should_ be removed and the error fixed upstream
                    # for reference, a test that would break is
                    # yt/data_objects/tests/test_fluxes.py::ExporterTests
                    pass
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
            units = self.ds.stream_handler.field_units[field]
            if units != "":
                self.add_output_field(field, sampling_type="cell", units=units)

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
        name = f"InMemoryParameterFile_{uuid.uuid4().hex}"

        _cached_datasets[name] = self
        Dataset.__init__(
            self,
            name,
            self._dataset_type,
            unit_system="cgs",
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
