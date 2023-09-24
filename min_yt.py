from __future__ import annotations

import os
import weakref
from collections import UserDict, defaultdict
from itertools import chain

import numpy as np
from yt.utilities.lib.misc_utilities import obtain_relative_velocity_vector


class YTFieldNotFound(Exception):
    pass


class CartesianCoordinateHandler:
    def __init__(self, ds):
        self.ds = weakref.proxy(ds)

    def setup_fields(self, registry):
        def _get_vert_fields(axi, units="cm"):
            def _vert(field, data):
                fcoords_vertex = np.random.random((data.nd, data.nd, data.nd, 8, 3))
                rv = np.array(fcoords_vertex[..., axi])
                return rv

            return _vert

        for axi, ax in enumerate("xyz"):
            f3 = _get_vert_fields(axi)
            registry.add_field(
                ("index", f"vertex_{ax}"),
                sampling_type="cell",
                function=f3,
                display_field=False,
                units="cm",
            )


class FieldDetector(defaultdict):
    def __init__(self, nd=16, ds=None):
        self.nd = nd
        self.field_parameters = {}
        self.ds = ds

        class fake_index:
            class fake_io:
                pass

            io = fake_io()

        self.index = fake_index()
        self.requested = []
        self.requested_parameters = []
        super().__init__(
            lambda: np.ones((nd, nd, nd), dtype="float64")
            + 1e-4 * np.random.random((nd, nd, nd))
        )

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

    def __call__(self, data):
        """Return the value of the field in a given *data* object."""
        return self._function(self, data)


class StreamDictFieldHandler(UserDict):
    @property
    def all_fields(self):
        fields = chain.from_iterable(s.keys() for s in self.values())
        return list(set(fields))


class StreamHandler:
    def __init__(self):
        self.fields = StreamDictFieldHandler()

    def get_fields(self):
        return self.fields.all_fields


class FieldInfoContainer(UserDict):
    known_other_fields = (("density", ("g/cm**3", ["density"], None)),)
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
        function,
        sampling_type: str,
        *,
        alias=None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("ds", self.ds)
        self[name] = DerivedField(name, sampling_type, function, alias=alias, **kwargs)

    def load_all_plugins(self, ftype: str | None = "gas") -> None:
        loaded = []
        orig = set(self.items())
        setup_fluid_fields(self, ftype, slice_info=self.slice_info)
        loaded += [n for n, v in set(self.items()).difference(orig)]

        deps, unavailable = self.check_derived_fields(loaded)
        self.ds.field_dependencies.update(deps)
        # Note we may have duplicated
        dfl = set(self.ds.derived_field_list).union(deps.keys())
        self.ds.derived_field_list = sorted(dfl)
        return loaded, unavailable

    def alias(self, alias_name, original_name):
        self.field_aliases[alias_name] = original_name

        def _TranslationFunc(field, data):
            return data[original_name].copy()

        _TranslationFunc.alias_name = original_name
        function = _TranslationFunc

        self.add_field(
            alias_name,
            function=function,
            sampling_type=self[original_name].sampling_type,
            display_name=self[original_name].display_name,
            units="cm",
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
        set(self.ds.derived_field_list).union(deps.keys())

        self.ds.derived_field_list = []
        return deps, unavailable


class Index:
    def __init__(self, ds):
        self.stream_handler = StreamHandler()
        self.dataset = weakref.proxy(ds)
        self.ds = self.dataset
        self.field_list = list(set(self.stream_handler.get_fields()))


class Dataset:
    default_fluid_type = "gas"
    fluid_types = ("gas", "deposit", "index", "stream")

    def __init__(self):
        self.coordinates = CartesianCoordinateHandler(self)

    def _get_field_info(self, field, /):
        ftype, fname = field
        if (ftype, fname) in self.field_info:
            return self.field_info[ftype, fname]
        else:
            raise YTFieldNotFound


def setup_fluid_fields(registry, ftype="gas", slice_info=None):
    def create_vector_fields(registry) -> None:
        def foo_closure(field, data):
            obtain_relative_velocity_vector(data, (xn, yn, zn), "bulk_velocity")

        xn, yn, zn = (("gas", f"velocity_{ax}") for ax in "xyz")

        registry.add_field(
            ("gas", "velocity_spherical_radius"),
            sampling_type="local",
            function=foo_closure,
            units="cm/s",
        )

    create_vector_fields(registry)
