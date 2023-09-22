from __future__ import annotations

import abc
import functools
import hashlib
import itertools
import os
import sys
import time
import uuid
import weakref
from collections import UserDict, defaultdict
from contextlib import contextmanager
from functools import cached_property
from importlib.util import find_spec
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import unyt as un
import yt.geometry
from more_itertools import always_iterable, unzip
from sympy import Symbol
from typing_extensions import assert_never
from unyt import Unit, UnitSystem, unyt_quantity
from unyt.exceptions import UnitConversionError, UnitParseError
from yt._maintenance.deprecation import issue_deprecation_warning
from yt._typing import AnyFieldKey, FieldKey, FieldName
from yt.config import ytcfg
from yt.data_objects.data_containers import YTDataContainer
from yt.data_objects.derived_quantities import DerivedQuantityCollection
from yt.data_objects.field_data import YTFieldData
from yt.data_objects.particle_filters import ParticleFilter, filter_registry
from yt.data_objects.profiles import create_profile
from yt.data_objects.region_expression import RegionExpression
from yt.data_objects.selection_objects.data_selection_objects import (
    YTSelectionContainer3D,
)
from yt.data_objects.static_output import _cached_datasets, _ds_store
from yt.data_objects.unions import ParticleUnion
from yt.fields.derived_field import DerivedField, ValidateSpatial
from yt.fields.field_exceptions import NeedsGridType
from yt.fields.field_type_container import FieldTypeContainer
from yt.fields.fluid_fields import setup_gradient_fields
from yt.frontends.stream.api import StreamFieldInfo, StreamHierarchy
from yt.frontends.ytdata.utilities import save_as_dataset
from yt.funcs import (
    get_output_filename,
    iter_fields,
    mylog,
    parse_center_array,
    set_intersection,
    validate_3d_array,
    validate_center,
    validate_object,
    validate_sequence,
)
from yt.geometry.api import Geometry
from yt.geometry.coordinates.api import (
    CartesianCoordinateHandler,
    CoordinateHandler,
    CylindricalCoordinateHandler,
    GeographicCoordinateHandler,
    InternalGeographicCoordinateHandler,
    PolarCoordinateHandler,
    SpectralCubeCoordinateHandler,
    SphericalCoordinateHandler,
)
from yt.geometry.geometry_handler import Index
from yt.geometry.selection_routines import compose_selector
from yt.units import UnitContainer, _wrap_display_ytarray, dimensions
from yt.units._numpy_wrapper_functions import uconcatenate
from yt.units.dimensions import current_mks
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
    YTCouldNotGenerateField,
    YTDataSelectorNotImplemented,
    YTDimensionalityError,
    YTException,
    YTFieldNotFound,
    YTFieldNotParseable,
    YTFieldTypeNotFound,
    YTFieldUnitError,
    YTFieldUnitParseError,
    YTIllDefinedParticleFilter,
    YTNonIndexedDataContainer,
    YTSpatialFieldUnitError,
)
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.object_registries import data_object_registry
from yt.utilities.on_demand_imports import _firefly as firefly
from yt.utilities.parallel_tools.parallel_analysis_interface import (
    ParallelAnalysisInterface,
    parallel_root_only,
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

        self._current_particle_type = "all"
        self._current_fluid_type = self.ds.default_fluid_type
        self.ds.objects.append(weakref.proxy(self))
        mylog.debug("Appending object to %s (type: %s)", self.ds, type(self))
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
    def pf(self):
        return getattr(self, "ds", None)

    @property
    def index(self):
        if self._index is not None:
            return self._index
        self._index = self.ds.index
        return self._index

    def _debug(self):
        """
        When called from within a derived field, this will run pdb.  However,
        during field detection, it will not.  This allows you to more easily
        debug fields that are being called on actual objects.
        """
        import pdb

        pdb.set_trace()

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
            axis = getattr(self, "axis", None)
            self.center = parse_center_array(center, ds=self.ds, axis=axis)
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

    def has_key(self, key):
        """
        Checks if a data field already exists.
        """
        return key in self.field_data

    def keys(self):
        return self.field_data.keys()

    def _reshape_vals(self, arr):
        return arr

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

    def _generate_particle_field(self, field):
        # First we check the validator
        ftype, fname = field
        if self._current_chunk is None or self._current_chunk.chunk_type != "spatial":
            gen_obj = self
        else:
            gen_obj = self._current_chunk.objs[0]
        try:
            finfo = self.ds._get_field_info(field)
            finfo.check_available(gen_obj)
        except NeedsGridType as ngt_exception:
            if ngt_exception.ghost_zones != 0:
                raise NotImplementedError from ngt_exception
            size = self._count_particles(ftype)
            rv = self.ds.arr(np.empty(size, dtype="float64"), finfo.units)
            ind = 0
            for _io_chunk in self.chunks([], "io", cache=False):
                for _chunk in self.chunks(field, "spatial"):
                    x, y, z = (self[ftype, f"particle_position_{ax}"] for ax in "xyz")
                    if x.size == 0:
                        continue
                    mask = self._current_chunk.objs[0].select_particles(
                        self.selector, x, y, z
                    )
                    if mask is None:
                        continue
                    # This requests it from the grid and does NOT mask it
                    data = self[field][mask]
                    rv[ind : ind + data.size] = data
                    ind += data.size
        else:
            with self._field_type_state(ftype, finfo, gen_obj):
                rv = self.ds._get_field_info(field)(gen_obj)
        return rv

    def _count_particles(self, ftype):
        for (f1, _f2), val in self.field_data.items():
            if f1 == ftype:
                return val.size
        size = 0
        for _io_chunk in self.chunks([], "io", cache=False):
            for _chunk in self.chunks([], "spatial"):
                x, y, z = (self[ftype, f"particle_position_{ax}"] for ax in "xyz")
                if x.size == 0:
                    continue
                size += self._current_chunk.objs[0].count_particles(
                    self.selector, x, y, z
                )
        return size

    def _generate_container_field(self, field):
        raise NotImplementedError

    def _parameter_iterate(self, seq):
        for obj in seq:
            old_fp = obj.field_parameters
            obj.field_parameters = self.field_parameters
            yield obj
            obj.field_parameters = old_fp

    def write_out(self, filename, fields=None, format="%0.16e"):
        """Write out the YTDataContainer object in a text file.

        This function will take a data object and produce a tab delimited text
        file containing the fields presently existing and the fields given in
        the ``fields`` list.

        Parameters
        ----------
        filename : String
            The name of the file to write to.

        fields : List of string, Default = None
            If this is supplied, these fields will be added to the list of
            fields to be saved to disk. If not supplied, whatever fields
            presently exist will be used.

        format : String, Default = "%0.16e"
            Format of numbers to be written in the file.

        Raises
        ------
        ValueError
            Raised when there is no existing field.

        YTException
            Raised when field_type of supplied fields is inconsistent with the
            field_type of existing fields.

        Examples
        --------
        >>> ds = fake_particle_ds()
        >>> sp = ds.sphere(ds.domain_center, 0.25)
        >>> sp.write_out("sphere_1.txt")
        >>> sp.write_out("sphere_2.txt", fields=["cell_volume"])
        """
        if fields is None:
            fields = sorted(self.field_data.keys())

        field_order = [("index", k) for k in self._key_fields]
        diff_fields = [field for field in fields if field not in field_order]
        field_order += diff_fields
        field_order = sorted(self._determine_fields(field_order))

        field_shapes = defaultdict(list)
        for field in field_order:
            shape = self[field].shape
            field_shapes[shape].append(field)

        # Check all fields have the same shape
        if len(field_shapes) != 1:
            err_msg = ["Got fields with different number of elements:\n"]
            for shape, these_fields in field_shapes.items():
                err_msg.append(f"\t {these_fields} with shape {shape}")
            raise YTException("\n".join(err_msg))

        with open(filename, "w") as fid:
            field_header = [str(f) for f in field_order]
            fid.write("\t".join(["#"] + field_header + ["\n"]))
            field_data = np.array([self.field_data[field] for field in field_order])
            for line in range(field_data.shape[1]):
                field_data[:, line].tofile(fid, sep="\t", format=format)
                fid.write("\n")

    def to_dataframe(self, fields):
        r"""Export a data object to a :class:`~pandas.DataFrame`.

        This function will take a data object and an optional list of fields
        and export them to a :class:`~pandas.DataFrame` object.
        If pandas is not importable, this will raise ImportError.

        Parameters
        ----------
        fields : list of strings or tuple field names
            This is the list of fields to be exported into
            the DataFrame.

        Returns
        -------
        df : :class:`~pandas.DataFrame`
            The data contained in the object.

        Examples
        --------
        >>> dd = ds.all_data()
        >>> df = dd.to_dataframe([("gas", "density"), ("gas", "temperature")])
        """
        from yt.utilities.on_demand_imports import _pandas as pd

        data = {}
        fields = self._determine_fields(fields)
        for field in fields:
            data[field[-1]] = self[field]
        df = pd.DataFrame(data)
        return df

    def to_astropy_table(self, fields):
        """
        Export region data to a :class:~astropy.table.table.QTable,
        which is a Table object which is unit-aware. The QTable can then
        be exported to an ASCII file, FITS file, etc.

        See the AstroPy Table docs for more details:
        http://docs.astropy.org/en/stable/table/

        Parameters
        ----------
        fields : list of strings or tuple field names
            This is the list of fields to be exported into
            the QTable.

        Examples
        --------
        >>> sp = ds.sphere("c", (1.0, "Mpc"))
        >>> t = sp.to_astropy_table([("gas", "density"), ("gas", "temperature")])
        """
        from astropy.table import QTable

        t = QTable()
        fields = self._determine_fields(fields)
        for field in fields:
            t[field[-1]] = self[field].to_astropy()
        return t

    def save_as_dataset(self, filename=None, fields=None):
        r"""Export a data object to a reloadable yt dataset.

        This function will take a data object and output a dataset
        containing either the fields presently existing or fields
        given in the ``fields`` list.  The resulting dataset can be
        reloaded as a yt dataset.

        Parameters
        ----------
        filename : str, optional
            The name of the file to be written.  If None, the name
            will be a combination of the original dataset and the type
            of data container.
        fields : list of string or tuple field names, optional
            If this is supplied, it is the list of fields to be saved to
            disk.  If not supplied, all the fields that have been queried
            will be saved.

        Returns
        -------
        filename : str
            The name of the file that has been created.

        Examples
        --------

        >>> import yt
        >>> ds = yt.load("enzo_tiny_cosmology/DD0046/DD0046")
        >>> sp = ds.sphere(ds.domain_center, (10, "Mpc"))
        >>> fn = sp.save_as_dataset(fields=[("gas", "density"), ("gas", "temperature")])
        >>> sphere_ds = yt.load(fn)
        >>> # the original data container is available as the data attribute
        >>> print(sds.data[("gas", "density")])
        [  4.46237613e-32   4.86830178e-32   4.46335118e-32 ...,   6.43956165e-30
           3.57339907e-30   2.83150720e-30] g/cm**3
        >>> ad = sphere_ds.all_data()
        >>> print(ad[("gas", "temperature")])
        [  1.00000000e+00   1.00000000e+00   1.00000000e+00 ...,   4.40108359e+04
           4.54380547e+04   4.72560117e+04] K

        """

        keyword = f"{str(self.ds)}_{self._type_name}"
        filename = get_output_filename(filename, keyword, ".h5")

        data = {}
        if fields is not None:
            for f in self._determine_fields(fields):
                data[f] = self[f]
        else:
            data.update(self.field_data)
        # get the extra fields needed to reconstruct the container
        tds_fields = tuple(("index", t) for t in self._tds_fields)
        for f in [f for f in self._container_fields + tds_fields if f not in data]:
            data[f] = self[f]
        data_fields = list(data.keys())

        need_grid_positions = False
        need_particle_positions = False
        ptypes = []
        ftypes = {}
        for field in data_fields:
            if field in self._container_fields:
                ftypes[field] = "grid"
                need_grid_positions = True
            elif self.ds.field_info[field].sampling_type == "particle":
                if field[0] not in ptypes:
                    ptypes.append(field[0])
                ftypes[field] = field[0]
                need_particle_positions = True
            else:
                ftypes[field] = "grid"
                need_grid_positions = True
        # projections and slices use px and py, so don't need positions
        if self._type_name in ["cutting", "proj", "slice", "quad_proj"]:
            need_grid_positions = False

        if need_particle_positions:
            for ax in self.ds.coordinates.axis_order:
                for ptype in ptypes:
                    p_field = (ptype, f"particle_position_{ax}")
                    if p_field in self.ds.field_info and p_field not in data:
                        data_fields.append(field)
                        ftypes[p_field] = p_field[0]
                        data[p_field] = self[p_field]
        if need_grid_positions:
            for ax in self.ds.coordinates.axis_order:
                g_field = ("index", ax)
                if g_field in self.ds.field_info and g_field not in data:
                    data_fields.append(g_field)
                    ftypes[g_field] = "grid"
                    data[g_field] = self[g_field]
                g_field = ("index", "d" + ax)
                if g_field in self.ds.field_info and g_field not in data:
                    data_fields.append(g_field)
                    ftypes[g_field] = "grid"
                    data[g_field] = self[g_field]

        extra_attrs = {
            arg: getattr(self, arg, None) for arg in self._con_args + self._tds_attrs
        }
        extra_attrs["con_args"] = repr(self._con_args)
        extra_attrs["data_type"] = "yt_data_container"
        extra_attrs["container_type"] = self._type_name
        extra_attrs["dimensionality"] = self._dimensionality
        save_as_dataset(
            self.ds, filename, data, field_types=ftypes, extra_attrs=extra_attrs
        )

        return filename

    def to_glue(self, fields, label="yt", data_collection=None):
        """
        Takes specific *fields* in the container and exports them to
        Glue (http://glueviz.org) for interactive
        analysis. Optionally add a *label*. If you are already within
        the Glue environment, you can pass a *data_collection* object,
        otherwise Glue will be started.
        """
        from glue.core import Data, DataCollection

        if ytcfg.get("yt", "internals", "within_testing"):
            from glue.core.application_base import Application as GlueApplication
        else:
            try:
                from glue.app.qt.application import GlueApplication
            except ImportError:
                from glue.qt.glue_application import GlueApplication
        gdata = Data(label=label)
        for component_name in fields:
            gdata.add_component(self[component_name], component_name)

        if data_collection is None:
            dc = DataCollection([gdata])
            app = GlueApplication(dc)
            try:
                app.start()
            except AttributeError:
                # In testing we're using a dummy glue application object
                # that doesn't have a start method
                pass
        else:
            data_collection.append(gdata)

    def create_firefly_object(
        self,
        datadir=None,
        fields_to_include=None,
        fields_units=None,
        default_decimation_factor=100,
        velocity_units="km/s",
        coordinate_units="kpc",
        show_unused_fields=0,
        *,
        JSONdir=None,
        match_any_particle_types=True,
        **kwargs,
    ):
        r"""This function links a region of data stored in a yt dataset
        to the Python frontend API for [Firefly](http://github.com/ageller/Firefly),
        a browser-based particle visualization tool.

        Parameters
        ----------

        datadir : string
            Path to where any `.json` files should be saved. If a relative
            path will assume relative to `${HOME}`. A value of `None` will default to `${HOME}/Data`.

        fields_to_include : array_like of strings or field tuples
            A list of fields that you want to include in your
            Firefly visualization for on-the-fly filtering and
            colormapping.

        default_decimation_factor : integer
            The factor by which you want to decimate each particle group
            by (e.g. if there are 1e7 total particles in your simulation
            you might want to set this to 100 at first). Randomly samples
            your data like `shuffled_data[::decimation_factor]` so as to
            not overtax a system. This is adjustable on a per particle group
            basis by changing the returned reader's
            `reader.particleGroup[i].decimation_factor` before calling
            `reader.writeToDisk()`.

        velocity_units : string
            The units that the velocity should be converted to in order to
            show streamlines in Firefly. Defaults to km/s.

        coordinate_units : string
            The units that the coordinates should be converted to. Defaults to
            kpc.

        show_unused_fields : boolean
            A flag to optionally print the fields that are available, in the
            dataset but were not explicitly requested to be tracked.

        match_any_particle_types : boolean
            If True, when any of the fields_to_include match multiple particle
            groups then the field will be added for all matching particle
            groups. If False, an error is raised when encountering an ambiguous
            field. Default is True.

        Any additional keyword arguments are passed to
        firefly.data_reader.Reader.__init__

        Returns
        -------
        reader : Firefly.data_reader.Reader object
            A reader object from the Firefly, configured
            to output the current region selected

        Examples
        --------

            >>> ramses_ds = yt.load(
            ...     "/Users/agurvich/Desktop/yt_workshop/"
            ...     + "DICEGalaxyDisk_nonCosmological/output_00002/info_00002.txt"
            ... )

            >>> region = ramses_ds.sphere(ramses_ds.domain_center, (1000, "kpc"))

            >>> reader = region.create_firefly_object(
            ...     "IsoGalaxyRamses",
            ...     fields_to_include=[
            ...         "particle_extra_field_1",
            ...         "particle_extra_field_2",
            ...     ],
            ...     fields_units=["dimensionless", "dimensionless"],
            ... )

            >>> reader.settings["color"]["io"] = [1, 1, 0, 1]
            >>> reader.particleGroups[0].decimation_factor = 100
            >>> reader.writeToDisk()
        """

        ## handle default arguments
        if fields_to_include is None:
            fields_to_include = []
        if fields_units is None:
            fields_units = []

        ## handle input validation, if any
        if len(fields_units) != len(fields_to_include):
            raise RuntimeError("Each requested field must have units.")

        ## for safety, in case someone passes a float just cast it
        default_decimation_factor = int(default_decimation_factor)

        if JSONdir is not None:
            issue_deprecation_warning(
                "The 'JSONdir' keyword argument is a deprecated alias for 'datadir'."
                "Please use 'datadir' directly.",
                stacklevel=3,
                since="4.1",
            )
            datadir = JSONdir

        ## initialize a firefly reader instance
        reader = firefly.data_reader.Reader(
            datadir=datadir, clean_datadir=True, **kwargs
        )

        ## Ensure at least one field type contains every field requested
        if match_any_particle_types:
            # Need to keep previous behavior: single string field names that
            # are ambiguous should bring in any matching ParticleGroups instead
            # of raising an error
            # This can be expanded/changed in the future to include field
            # tuples containing some sort of special "any" ParticleGroup
            unambiguous_fields_to_include = []
            unambiguous_fields_units = []
            for field, field_unit in zip(fields_to_include, fields_units):
                if isinstance(field, tuple):
                    # skip tuples, they'll be checked with _determine_fields
                    unambiguous_fields_to_include.append(field)
                    unambiguous_fields_units.append(field_unit)
                    continue
                _, candidates = self.ds._get_field_info_helper(field)
                if len(candidates) == 1:
                    # Field is unambiguous, add in tuple form
                    # This should be equivalent to _tupleize_field
                    unambiguous_fields_to_include.append(candidates[0])
                    unambiguous_fields_units.append(field_unit)
                else:
                    # Field has multiple candidates, add all of them instead
                    # of original field. Note this may bring in aliases and
                    # equivalent particle fields
                    for c in candidates:
                        unambiguous_fields_to_include.append(c)
                        unambiguous_fields_units.append(field_unit)
            fields_to_include = unambiguous_fields_to_include
            fields_units = unambiguous_fields_units
        # error if any requested field is unknown or (still) ambiguous
        # This is also sufficient if match_any_particle_types=False
        fields_to_include = self._determine_fields(fields_to_include)
        ## Also generate equivalent of particle_fields_by_type including
        ## derived fields
        kysd = defaultdict(list)
        for k, v in self.ds.derived_field_list:
            kysd[k].append(v)

        ## create a ParticleGroup object that contains *every* field
        for ptype in sorted(self.ds.particle_types_raw):
            ## skip this particle type if it has no particles in this dataset
            if self[ptype, "relative_particle_position"].shape[0] == 0:
                continue

            ## loop through the fields and print them to the screen
            if show_unused_fields:
                ## read the available extra fields from yt
                this_ptype_fields = self.ds.particle_fields_by_type[ptype]

                ## load the extra fields and print them
                for field in this_ptype_fields:
                    if field not in fields_to_include:
                        mylog.warning(
                            "detected (but did not request) %s %s", ptype, field
                        )

            field_arrays = []
            field_names = []

            ## explicitly go after the fields we want
            for field, units in zip(fields_to_include, fields_units):
                ## Only interested in fields with the current particle type,
                ## whether that means general fields or field tuples
                ftype, fname = field
                if ftype not in (ptype, "all"):
                    continue

                ## determine if you want to take the log of the field for Firefly
                log_flag = "log(" in units

                ## read the field array from the dataset
                this_field_array = self[ptype, fname]

                ## fix the units string and prepend 'log' to the field for
                ##  the UI name
                if log_flag:
                    units = units[len("log(") : -1]
                    fname = f"log{fname}"

                ## perform the unit conversion and take the log if
                ##  necessary.
                this_field_array.in_units(units)
                if log_flag:
                    this_field_array = np.log10(this_field_array)

                ## add this array to the tracked arrays
                field_arrays += [this_field_array]
                field_names = np.append(field_names, [fname], axis=0)

            ## flag whether we want to filter and/or color by these fields
            ##  we'll assume yes for both cases, this can be changed after
            ##  the reader object is returned to the user.
            field_filter_flags = np.ones(len(field_names))
            field_colormap_flags = np.ones(len(field_names))

            ## field_* needs to be explicitly set None if empty
            ## so that Firefly will correctly compute the binary
            ## headers
            if len(field_arrays) == 0:
                if len(fields_to_include) > 0:
                    mylog.warning("No additional fields specified for %s", ptype)
                field_arrays = None
                field_names = None
                field_filter_flags = None
                field_colormap_flags = None

            ## Check if particles have velocities
            if "relative_particle_velocity" in kysd[ptype]:
                velocities = self[ptype, "relative_particle_velocity"].in_units(
                    velocity_units
                )
            else:
                velocities = None

            ## create a firefly ParticleGroup for this particle type
            pg = firefly.data_reader.ParticleGroup(
                UIname=ptype,
                coordinates=self[ptype, "relative_particle_position"].in_units(
                    coordinate_units
                ),
                velocities=velocities,
                field_arrays=field_arrays,
                field_names=field_names,
                field_filter_flags=field_filter_flags,
                field_colormap_flags=field_colormap_flags,
                decimation_factor=default_decimation_factor,
            )

            ## bind this particle group to the firefly reader object
            reader.addParticleGroup(pg)

        return reader

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
            weight_field = sanitize_weight_field( weight)
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
        old_particle_type = obj._current_particle_type
        old_fluid_type = obj._current_fluid_type
        fluid_types = self.ds.fluid_types
        if finfo.sampling_type == "particle" and ftype not in fluid_types:
            obj._current_particle_type = ftype
        else:
            obj._current_fluid_type = ftype
        yield
        obj._current_particle_type = old_particle_type
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

            # these tests are really insufficient as a field type may be valid, and the
            # field name may be valid, but not the combination (field type, field name)
            particle_field = finfo.sampling_type == "particle"
            local_field = finfo.local_sampling
            if local_field:
                pass
            elif particle_field and ftype not in self.ds.particle_types:
                raise YTFieldTypeNotFound(ftype, ds=self.ds)
            elif not particle_field and ftype not in self.ds.fluid_types:
                raise YTFieldTypeNotFound(ftype, ds=self.ds)
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
        apply_fields = defaultdict(list)
        for field in self._determine_fields(fields):
            # We need to create the field on the raw particle types
            # for particles types (when the field is not directly
            # defined for the derived particle type only)
            finfo = self.ds.field_info[field]

            nfields.append(field)
        for filter_type in apply_fields:
            f = self.ds.known_filters[filter_type]
            with f.apply(self):
                self.get_data(apply_fields[filter_type])
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
        fluids, particles = [], []
        finfos = {}
        for field_key in fields_to_get:
            finfo = self.ds._get_field_info(field_key)
            finfos[field_key] = finfo
            if finfo.sampling_type == "particle":
                particles.append(field_key)
            elif field_key not in fluids:
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

        read_particles, gen_particles = self.index._read_particle_fields(
            particles, self, self._current_chunk
        )

        for f, v in read_particles.items():
            self.field_data[f] = self.ds.arr(v, units=finfos[f].units)
            self.field_data[f].convert_to_units(finfos[f].output_units)

        fields_to_generate += gen_fluids + gen_particles
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

                        if fi.dimensions is None:
                            mylog.warning(
                                "Field %s was added without specifying units or dimensions, "
                                "auto setting units to %r",
                                fi.name,
                                sunits or "dimensionless",
                            )
                        elif not dimensions_compare_equal(fi.dimensions, dimensions):
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
        geometry: Geometry = self.ds.geometry
        if geometry is Geometry.CARTESIAN:
            le, re = self._get_bbox()
            le.convert_to_units("code_length")
            re.convert_to_units("code_length")
            return le, re
        elif (
            geometry is Geometry.CYLINDRICAL
            or geometry is Geometry.POLAR
            or geometry is Geometry.SPHERICAL
            or geometry is Geometry.GEOGRAPHIC
            or geometry is Geometry.INTERNAL_GEOGRAPHIC
            or geometry is Geometry.SPECTRAL_CUBE
        ):
            raise NotImplementedError(
                f"get_bbox is currently not implemented for {geometry=}!"
            )
        else:
            assert_never(geometry)

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
        if center is not None:
            validate_center(center)
        validate_3d_array(left_edge)
        validate_3d_array(right_edge)
        validate_sequence(fields)
        validate_object(field_parameters, dict)
        validate_object(data_source, YTSelectionContainer)
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
        self.particle_types = {}
        self.periodicity = periodicity
        self.cell_widths = cell_widths
        self.parameters = {}

    def get_fields(self):
        return self.fields.all_fields


class Dataset(abc.ABC):
    _load_requirements: list[str] = []
    default_fluid_type = "gas"
    default_field = ("gas", "density")
    fluid_types: tuple[FieldType, ...] = ("gas", "deposit", "index")
    particle_types: tuple[ParticleType, ...] = ("io",)  # By default we have an 'all'
    particle_types_raw: Optional[tuple[ParticleType, ...]] = ("io",)
    geometry: Geometry = Geometry.CARTESIAN
    coordinates = None
    storage_filename = None
    particle_unions: Optional[dict[ParticleType, ParticleUnion]] = None
    known_filters: Optional[dict[ParticleType, ParticleFilter]] = None
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
        self.known_filters = self.known_filters or {}
        self.particle_unions = self.particle_unions or {}
        self.field_units = self.field_units or {}
        self._determined_fields = {}
        self.units_override = self.__class__._sanitize_units_override(units_override)
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

    def get_smallest_appropriate_unit(
        self, v, quantity="distance", return_quantity=False
    ):
        """
        Returns the largest whole unit smaller than the YTQuantity passed to
        it as a string.

        The quantity keyword can be equal to `distance` or `time`.  In the
        case of distance, the units are: 'Mpc', 'kpc', 'pc', 'au', 'rsun',
        'km', etc.  For time, the units are: 'Myr', 'kyr', 'yr', 'day', 'hr',
        's', 'ms', etc.

        If return_quantity is set to True, it finds the largest YTQuantity
        object with a whole unit and a power of ten as the coefficient, and it
        returns this YTQuantity.
        """
        good_u = None
        if quantity == "distance":
            unit_list = [
                "Ppc",
                "Tpc",
                "Gpc",
                "Mpc",
                "kpc",
                "pc",
                "au",
                "rsun",
                "km",
                "m",
                "cm",
                "mm",
                "um",
                "nm",
                "pm",
            ]
        elif quantity == "time":
            unit_list = [
                "Yyr",
                "Zyr",
                "Eyr",
                "Pyr",
                "Tyr",
                "Gyr",
                "Myr",
                "kyr",
                "yr",
                "day",
                "hr",
                "s",
                "ms",
                "us",
                "ns",
                "ps",
                "fs",
            ]
        else:
            raise ValueError(
                "Specified quantity must be equal to 'distance' or 'time'."
            )
        for unit in unit_list:
            uq = self.quan(1.0, unit)
            if uq <= v:
                good_u = unit
                break
        if good_u is None and quantity == "distance":
            good_u = "cm"
        if good_u is None and quantity == "time":
            good_u = "s"
        if return_quantity:
            unit_index = unit_list.index(good_u)
            # This avoids indexing errors
            if unit_index == 0:
                return self.quan(1, unit_list[0])
            # Number of orders of magnitude between unit and next one up
            OOMs = np.ceil(
                np.log10(
                    self.quan(1, unit_list[unit_index - 1])
                    / self.quan(1, unit_list[unit_index])
                )
            )
            # Backwards order of coefficients (e.g. [100, 10, 1])
            coeffs = 10 ** np.arange(OOMs)[::-1]
            for j in coeffs:
                uq = self.quan(j, good_u)
                if uq <= v:
                    return uq
        else:
            return good_u

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
        for ptype in self.particle_types:
            self.field_info.setup_particle_fields(ptype)
        self.field_info.setup_fluid_index_fields()
        if "all" not in self.particle_types:
            pu = ParticleUnion("all", list(self.particle_types_raw))
            self.add_particle_union(pu)
        if "nbody" not in self.particle_types:
            ptypes = list(self.particle_types_raw)
            if hasattr(self, "_sph_ptypes"):
                for sph_ptype in self._sph_ptypes:
                    if sph_ptype in ptypes:
                        ptypes.remove(sph_ptype)
            if ptypes:
                nbody_ptypes = []
                for ptype in ptypes:
                    if (ptype, "particle_mass") in self.field_info:
                        nbody_ptypes.append(ptype)
                pu = ParticleUnion("nbody", nbody_ptypes)
                self.add_particle_union(pu)
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

    def setup_deprecated_fields(self):
        from yt.fields.field_aliases import _field_name_aliases

        added = []
        for old_name, new_name in _field_name_aliases:
            try:
                fi = self._get_field_info(new_name)
            except YTFieldNotFound:
                continue
            self.field_info.alias(("gas", old_name), fi.name)
            added.append(("gas", old_name))
        self.field_info.find_dependencies(added)

    def _setup_coordinate_handler(self, axis_order: Optional[AxisOrder]) -> None:
        # backward compatibility layer:
        # turning off type-checker on a per-line basis
        cls: type[CoordinateHandler]

        # end compatibility layer
        if not isinstance(self.geometry, Geometry):
            raise TypeError(
                "Expected dataset.geometry attribute to be of "
                "type yt.geometry.geometry_enum.Geometry\n"
                f"Got {self.geometry=} with type {type(self.geometry)}"
            )

        if self.geometry is Geometry.CARTESIAN:
            cls = CartesianCoordinateHandler
        elif self.geometry is Geometry.CYLINDRICAL:
            cls = CylindricalCoordinateHandler
        elif self.geometry is Geometry.POLAR:
            cls = PolarCoordinateHandler
        elif self.geometry is Geometry.SPHERICAL:
            cls = SphericalCoordinateHandler
            # It shouldn't be required to reset self.no_cgs_equiv_length
            # to the default value (False) here, but it's still necessary
            # see https://github.com/yt-project/yt/pull/3618
            self.no_cgs_equiv_length = False
        elif self.geometry is Geometry.GEOGRAPHIC:
            cls = GeographicCoordinateHandler
            self.no_cgs_equiv_length = True
        elif self.geometry is Geometry.INTERNAL_GEOGRAPHIC:
            cls = InternalGeographicCoordinateHandler
            self.no_cgs_equiv_length = True
        elif self.geometry is Geometry.SPECTRAL_CUBE:
            cls = SpectralCubeCoordinateHandler
        else:
            assert_never(self.geometry)

        self.coordinates = cls(self, ordering=axis_order)

    def add_particle_union(self, union):
        # No string lookups here, we need an actual union.
        f = self.particle_fields_by_type

        # find fields common to all particle types in the union
        fields = set_intersection([f[s] for s in union if s in self.particle_types_raw])

        if len(fields) == 0:
            # don't create this union if no fields are common to all
            # particle types
            return len(fields)

        for field in fields:
            units = set()
            for s in union:
                # First we check our existing fields for units
                funits = self._get_field_info((s, field)).units
                # Then we override with field_units settings.
                funits = self.field_units.get((s, field), funits)
                units.add(funits)
            if len(units) == 1:
                self.field_units[union.name, field] = list(units)[0]
        self.particle_types += (union.name,)
        self.particle_unions[union.name] = union
        fields = [(union.name, field) for field in fields]
        new_fields = [_ for _ in fields if _ not in self.field_list]
        self.field_list.extend(new_fields)
        new_field_info_fields = [
            _ for _ in fields if _ not in self.field_info.field_list
        ]
        self.field_info.field_list.extend(new_field_info_fields)
        self.index.field_list = sorted(self.field_list)
        # Give ourselves a chance to add them here, first, then...
        # ...if we can't find them, we set them up as defaults.
        new_fields = self._setup_particle_types([union.name])
        self.field_info.find_dependencies(new_fields)
        return len(new_fields)

    def add_particle_filter(self, filter):
        """Add particle filter to the dataset.

        Add ``filter`` to the dataset and set up relevant derived_field.
        It will also add any ``filtered_type`` that the ``filter`` depends on.

        """
        # This requires an index
        self.index
        # This is a dummy, which we set up to enable passthrough of "all"
        # concatenation fields.
        n = getattr(filter, "name", filter)
        self.known_filters[n] = None
        if isinstance(filter, str):
            used = False
            f = filter_registry.get(filter, None)
            if f is None:
                return False
            used = self._setup_filtered_type(f)
            if used:
                filter = f
        else:
            used = self._setup_filtered_type(filter)
        if not used:
            self.known_filters.pop(n, None)
            return False
        self.known_filters[filter.name] = filter
        return True

    def _setup_filtered_type(self, filter):
        # Check if the filtered_type of this filter is known,
        # otherwise add it first if it is in the filter_registry
        if filter.filtered_type not in self.known_filters.keys():
            if filter.filtered_type in filter_registry:
                self.add_particle_filter(filter.filtered_type)

        if not filter.available(self.derived_field_list):
            raise YTIllDefinedParticleFilter(
                filter, filter.missing(self.derived_field_list)
            )
        fi = self.field_info
        fd = self.field_dependencies
        available = False
        for fn in self.derived_field_list:
            if fn[0] == filter.filtered_type:
                # Now we can add this
                available = True
                self.derived_field_list.append((filter.name, fn[1]))
                fi[filter.name, fn[1]] = filter.wrap_func(fn, fi[fn])
                # Now we append the dependencies
                fd[filter.name, fn[1]] = fd[fn]
        if available:
            if filter.name not in self.particle_types:
                self.particle_types += (filter.name,)
            if hasattr(self, "_sph_ptypes"):
                if filter.filtered_type in (self._sph_ptypes + ("gas",)):
                    self._sph_ptypes = self._sph_ptypes + (filter.name,)
            new_fields = self._setup_particle_types([filter.name])
            deps, _ = self.field_info.check_derived_fields(new_fields)
            self.field_dependencies.update(deps)
        return available

    def _setup_particle_types(self, ptypes=None):
        df = []
        if ptypes is None:
            ptypes = self.ds.particle_types_raw
        for ptype in set(ptypes):
            df += self._setup_particle_type(ptype)
        return df

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

        def _are_ambiguous(candidates: list[FieldKey]) -> bool:
            if len(candidates) < 2:
                return False

            ftypes, fnames = (list(_) for _ in unzip(candidates))
            assert all(name == fnames[0] for name in fnames)

            fi = self.field_info

            all_aliases: bool = all(
                fi[c].is_alias_to(fi[candidates[0]]) for c in candidates
            )

            all_equivalent_particle_fields: bool
            if (
                not self.particle_types
                or not self.particle_unions
                or not self.particle_types_raw
            ):
                all_equivalent_particle_fields = False
            elif all(ft in self.particle_types for ft in ftypes):
                ptypes = ftypes

                sub_types_list: list[set[str]] = []
                for pt in ptypes:
                    if pt in self.particle_types_raw:
                        sub_types_list.append({pt})
                    elif pt in self.particle_unions:
                        sub_types_list.append(set(self.particle_unions[pt].sub_types))
                all_equivalent_particle_fields = all(
                    st == sub_types_list[0] for st in sub_types_list
                )
            else:
                all_equivalent_particle_fields = False

            return not (all_aliases or all_equivalent_particle_fields)

        if _are_ambiguous(candidates):
            ft, fn = field_info.name
            possible_ftypes = [c[0] for c in candidates]
            raise ValueError(
                f"The requested field name {fn!r} "
                "is ambiguous and corresponds to any one of "
                f"the following field types:\n {possible_ftypes}\n"
                "Please specify the requested field as an explicit "
                "tuple (<ftype>, <fname>).\n"
            )
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

        if ftype == "unknown":
            candidates: list[FieldKey] = [
                (ft, fn) for ft, fn in self.field_info if fn == fname
            ]

            # We also should check "all" for particles, which can show up if you're
            # mixing deposition/gas fields with particle fields.
            if hasattr(self, "_sph_ptype"):
                to_guess = [self.default_fluid_type, "all"]
            else:
                to_guess = ["all", self.default_fluid_type]
            to_guess += list(self.fluid_types) + list(self.particle_types)
            for ftype in to_guess:
                if (ftype, fname) in self.field_info:
                    return self.field_info[ftype, fname], candidates

        elif (ftype, fname) in self.field_info:
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
    def particle_fields_by_type(self):
        fields = defaultdict(list)
        for field in self.field_list:
            if field[0] in self.particle_types_raw:
                fields[field[0]].append(field[1])
        return fields

    @property
    def particles_exist(self):
        for pt, f in itertools.product(self.particle_types_raw, self.field_list):
            if pt == f[0]:
                return True
        return False

    @property
    def particle_type_counts(self):
        self.index
        if not self.particles_exist:
            return {}

        # frontends or index implementation can populate this dict while
        # creating the index if they know particle counts at that time
        if self._particle_type_counts is not None:
            return self._particle_type_counts

        self._particle_type_counts = self.index._get_particle_type_counts()
        return self._particle_type_counts

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
        # here we override units, if overrides have been provided.
        self._override_code_units()

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

    @classmethod
    def _validate_units_override_keys(cls, units_override):
        valid_keys = set(cls.default_units.keys())
        invalid_keys_found = set(units_override.keys()) - valid_keys
        if invalid_keys_found:
            raise ValueError(
                f"units_override contains invalid keys: {invalid_keys_found}"
            )

    default_units = {
        "length_unit": "cm",
        "time_unit": "s",
        "mass_unit": "g",
        "velocity_unit": "cm/s",
        "magnetic_unit": "gauss",
        "temperature_unit": "K",
    }

    @classmethod
    def _sanitize_units_override(cls, units_override):
        """
        Convert units_override values to valid input types for unyt.
        Throw meaningful errors early if units_override is ill-formed.

        Parameters
        ----------
        units_override : dict

            keys should be strings with format  "<dim>_unit" (e.g. "mass_unit"), and
            need to match a key in cls.default_units

            values should be mappable to unyt.unyt_quantity objects, and can be any
            combinations of:
                - unyt.unyt_quantity
                - 2-long sequence (tuples, list, ...) with types (number, str)
                  e.g. (10, "km"), (0.1, "s")
                - number (in which case the associated is taken from cls.default_unit)


        Raises
        ------
        TypeError
            If unit_override has invalid types

        ValueError
            If provided units do not match the intended dimensionality,
            or in case of a zero scaling factor.

        """
        uo = {}
        if units_override is None:
            return uo

        cls._validate_units_override_keys(units_override)

        for key in cls.default_units:
            try:
                val = units_override[key]
            except KeyError:
                continue

            # Now attempt to instantiate a unyt.unyt_quantity from val ...
            try:
                # ... directly (valid if val is a number, or a unyt_quantity)
                uo[key] = YTQuantity(val)
                continue
            except RuntimeError:
                # note that unyt.unyt_quantity throws RuntimeError in lieu of TypeError
                pass
            try:
                # ... with tuple unpacking (valid if val is a sequence)
                uo[key] = YTQuantity(*val)
                continue
            except (RuntimeError, TypeError, UnitParseError):
                pass
            raise TypeError(
                "units_override values should be 2-sequence (float, str), "
                "YTQuantity objects or real numbers; "
                f"received {val} with type {type(val)}."
            )
        for key, q in uo.items():
            if q.units.is_dimensionless:
                uo[key] = YTQuantity(q, cls.default_units[key])
            try:
                uo[key].to(cls.default_units[key])
            except UnitConversionError as err:
                raise ValueError(
                    "Inconsistent dimensionality in units_override. "
                    f"Received {key} = {uo[key]}"
                ) from err
            if uo[key].value == 0.0:
                raise ValueError(
                    f"Invalid 0 normalisation factor in units_override for {key}."
                )
        return uo

    def _override_code_units(self):
        if not self.units_override:
            return

        for ukey, val in self.units_override.items():
            setattr(self, ukey, self.quan(val))

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
        from yt.fields.field_functions import validate_field_function

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

    def add_mesh_sampling_particle_field(self, sample_field, ptype="all"):
        """Add a new mesh sampling particle field

        Creates a new particle field which has the value of the
        *deposit_field* at the location of each particle of type
        *ptype*.

        Parameters
        ----------

        sample_field : tuple
           The field name tuple of the mesh field to be deposited onto
           the particles. This must be a field name tuple so yt can
           appropriately infer the correct particle type.
        ptype : string, default 'all'
           The particle type onto which the deposition will occur.

        Returns
        -------

        The field name tuple for the newly created field.

        Examples
        --------
        >>> ds = yt.load("output_00080/info_00080.txt")
        ... ds.add_mesh_sampling_particle_field(("gas", "density"), ptype="all")

        >>> print("The density at the location of the particle is:")
        ... print(ds.r["all", "cell_gas_density"])
        The density at the location of the particle is:
        [9.33886124e-30 1.22174333e-28 1.20402333e-28 ... 2.77410331e-30
         8.79467609e-31 3.50665136e-30] g/cm**3

        >>> len(ds.r["all", "cell_gas_density"]) == len(ds.r["all", "particle_ones"])
        True

        """
        if isinstance(sample_field, tuple):
            ftype, sample_field = sample_field[0], sample_field[1]
        else:
            raise RuntimeError

        return self.index._add_mesh_sampling_particle_field(sample_field, ftype, ptype)

    def add_deposited_particle_field(
        self, deposit_field, method, kernel_name="cubic", weight_field=None
    ):
        """Add a new deposited particle field

        Creates a new deposited field based on the particle *deposit_field*.

        Parameters
        ----------

        deposit_field : tuple
           The field name tuple of the particle field the deposited field will
           be created from.  This must be a field name tuple so yt can
           appropriately infer the correct particle type.
        method : string
           This is the "method name" which will be looked up in the
           `particle_deposit` namespace as `methodname_deposit`.  Current
           methods include `simple_smooth`, `sum`, `std`, `cic`, `weighted_mean`,
           `nearest` and `count`.
        kernel_name : string, default 'cubic'
           This is the name of the smoothing kernel to use. It is only used for
           the `simple_smooth` method and is otherwise ignored. Current
           supported kernel names include `cubic`, `quartic`, `quintic`,
           `wendland2`, `wendland4`, and `wendland6`.
        weight_field : (field_type, field_name) or None
           Weighting field name for deposition method `weighted_mean`.
           If None, use the particle mass.

        Returns
        -------

        The field name tuple for the newly created field.
        """
        self.index
        if isinstance(deposit_field, tuple):
            ptype, deposit_field = deposit_field[0], deposit_field[1]
        else:
            raise RuntimeError

        if weight_field is None:
            weight_field = (ptype, "particle_mass")
        units = self.field_info[ptype, deposit_field].output_units
        take_log = self.field_info[ptype, deposit_field].take_log
        name_map = {
            "sum": "sum",
            "std": "std",
            "cic": "cic",
            "weighted_mean": "avg",
            "nearest": "nn",
            "simple_smooth": "ss",
            "count": "count",
        }
        field_name = "%s_" + name_map[method] + "_%s"
        field_name = field_name % (ptype, deposit_field.replace("particle_", ""))

        if method == "count":
            field_name = f"{ptype}_count"
            if ("deposit", field_name) in self.field_info:
                return ("deposit", field_name)
            else:
                units = "dimensionless"
                take_log = False

        def _deposit_field(field, data):
            """
            Create a grid field for particle quantities using given method.
            """
            pos = data[ptype, "particle_position"]
            fields = [data[ptype, deposit_field]]
            if method == "weighted_mean":
                fields.append(data[ptype, weight_field])
            fields = [np.ascontiguousarray(f) for f in fields]
            d = data.deposit(pos, fields, method=method, kernel_name=kernel_name)
            d = data.ds.arr(d, units=units)
            if method == "weighted_mean":
                d[np.isnan(d)] = 0.0
            return d

        self.add_field(
            ("deposit", field_name),
            function=_deposit_field,
            sampling_type="cell",
            units=units,
            take_log=take_log,
            validators=[ValidateSpatial()],
        )
        return ("deposit", field_name)

    def add_gradient_fields(self, fields=None):
        """Add gradient fields.

        Creates four new grid-based fields that represent the components of the gradient
        of an existing field, plus an extra field for the magnitude of the gradient. The
        gradient is computed using second-order centered differences.

        Parameters
        ----------
        fields : str or tuple(str, str), or a list of the previous
            Label(s) for at least one field. Can either represent a tuple
            (<field type>, <field fname>) or simply the field name.
            Warning: several field types may match the provided field name,
            in which case the first one discovered internally is used.

        Returns
        -------
        A list of field name tuples for the newly created fields.

        Raises
        ------
        YTFieldNotParsable
            If fields are not parsable to yt field keys.

        YTFieldNotFound :
            If at least one field can not be identified.

        Examples
        --------

        >>> grad_fields = ds.add_gradient_fields(("gas", "density"))
        >>> print(grad_fields)
        ... [
        ...     ("gas", "density_gradient_x"),
        ...     ("gas", "density_gradient_y"),
        ...     ("gas", "density_gradient_z"),
        ...     ("gas", "density_gradient_magnitude"),
        ... ]

        Note that the above example assumes ds.geometry == 'cartesian'. In general,
        the function will create gradient components along the axes of the dataset
        coordinate system.
        For instance, with cylindrical data, one gets 'density_gradient_<r,theta,z>'

        """
        if fields is None:
            raise TypeError("Missing required positional argument: fields")

        self.index
        data_obj = self.all_data()
        explicit_fields = data_obj._determine_fields(fields)
        grad_fields = []
        for ftype, fname in explicit_fields:
            units = self.field_info[ftype, fname].units
            setup_gradient_fields(self.field_info, (ftype, fname), units)
            # Now we make a list of the fields that were just made, to check them
            # and to return them
            grad_fields += [
                (ftype, f"{fname}_gradient_{suffix}")
                for suffix in self.coordinates.axis_order
            ]
            grad_fields.append((ftype, f"{fname}_gradient_magnitude"))
            deps, _ = self.field_info.check_derived_fields(grad_fields)
            self.field_dependencies.update(deps)
        return grad_fields

    _max_level = None

    @property
    def max_level(self):
        if self._max_level is None:
            self._max_level = self.index.max_level
        return self._max_level

    @max_level.setter
    def max_level(self, value):
        self._max_level = value

    _min_level = None

    @property
    def min_level(self):
        if self._min_level is None:
            self._min_level = self.index.min_level
        return self._min_level

    @min_level.setter
    def min_level(self, value):
        self._min_level = value

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


class MinimalStreamDataset(Dataset):
    _index_class = StreamHierarchy
    _field_info_class = StreamFieldInfo
    _dataset_type = "stream"

    def __init__(self, *, stream_handler):
        self.fluid_types += ("stream",)
        self.geometry = Geometry.CARTESIAN
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
