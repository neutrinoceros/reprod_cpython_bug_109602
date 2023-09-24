from __future__ import annotations

import weakref
from collections import UserDict, defaultdict

import numpy as np
from yt.utilities.lib.misc_utilities import obtain_relative_velocity_vector


class CartesianCoordinateHandler:
    def __init__(self, ds):
        self.ds = weakref.proxy(ds)

    def setup_fields(self, registry):
        def _get_vert_fields(axi):
            def _vert(field, data):
                fcoords_vertex = np.random.random((data.nd, data.nd, data.nd, 8, 3))
                rv = np.array(fcoords_vertex[..., axi])
                return rv

            return _vert

        for axi, ax in enumerate("xyz"):
            registry.add_field(
                ("index", f"vertex_{ax}"), function=_get_vert_fields(axi)
            )


class FieldDetector(defaultdict):
    def __init__(self, nd=16, ds=None):
        self.nd = nd
        self.field_parameters = {}
        self.ds = ds
        super().__init__(
            lambda: np.ones((nd, nd, nd), dtype="float64")
            + 1e-4 * np.random.random((nd, nd, nd))
        )

    def __missing__(self, field):
        finfo = self.ds._get_field_info(field)
        vv = finfo(self)
        self[finfo.name] = vv
        return vv


class DerivedField:
    def __init__(self, name, function, ds):
        self.name = name
        self.ds = ds
        self._function = function

    def get_dependencies(self, *args, **kwargs):
        """
        This returns a list of names of fields that this field depends on.
        """
        e = FieldDetector(*args, **kwargs)
        e[self.name]
        return e

    def __call__(self, data):
        """Return the value of the field in a given *data* object."""
        return self._function(self, data)


class FieldInfoContainer(UserDict):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def setup_fluid_index_fields(self):
        # Now we get all our index types and set up aliases to them
        index_fields = {f for _, f in self if _ == "index"}
        for f in index_fields:
            self.alias(("gas", f), ("index", f))

    def add_field(self, name, function, **kwargs):
        kwargs.setdefault("ds", self.ds)
        self[name] = DerivedField(name, function, **kwargs)

    def load_all_plugins(self) -> None:
        loaded = []
        orig = set(self.items())
        setup_fluid_fields(self)
        loaded += [n for n, v in set(self.items()).difference(orig)]

        deps, unavailable = self.check_derived_fields(loaded)
        self.ds.field_dependencies.update(deps)
        # Note we may have duplicated
        dfl = set(self.ds.derived_field_list).union(deps.keys())
        self.ds.derived_field_list = sorted(dfl)
        return loaded, unavailable

    def alias(self, alias_name, original_name):
        def _TranslationFunc(field, data):
            return data[original_name].copy()

        _TranslationFunc.alias_name = original_name

        self.add_field(alias_name, function=_TranslationFunc)

    def check_derived_fields(self, fields_to_check=None):
        deps = {}
        fields_to_check = fields_to_check or list(self.keys())
        for field in fields_to_check:
            fi = self[field]
            try:
                fd = fi.get_dependencies(ds=self.ds)
            except Exception:
                self.pop(field)
                continue
            deps[field] = fd

        self.ds.derived_field_list = []
        return deps, []


class Dataset:
    def __init__(self):
        self.coordinates = CartesianCoordinateHandler(self)

    def _get_field_info(self, field, /):
        ftype, fname = field
        if (ftype, fname) in self.field_info:
            return self.field_info[ftype, fname]
        else:
            raise Exception


def setup_fluid_fields(registry):
    def create_vector_fields(registry) -> None:
        def foo_closure(field, data):
            obtain_relative_velocity_vector(data, (xn, yn, zn), "bulk_velocity")

        xn, yn, zn = (("gas", f"velocity_{ax}") for ax in "xyz")

        registry.add_field(("gas", "velocity_spherical_radius"), function=foo_closure)

    create_vector_fields(registry)
