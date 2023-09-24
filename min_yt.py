from collections import UserDict, defaultdict

import numpy as np
from yt.utilities.lib.misc_utilities import obtain_relative_velocity_vector


class CartesianCoordinateHandler:
    def setup_fields(self, registry):
        def _get_vert_fields(axi):
            def _vert(field, data):
                fcoords_vertex = np.random.random((32, 32, 32, 8, 3))
                return np.array(fcoords_vertex[..., axi])

            return _vert

        registry.add_field(("index", "vertex_x"), function=_get_vert_fields(0))


class FieldDetector(defaultdict):
    def __init__(self, ds):
        self.ds = ds
        super().__init__(lambda: 1)

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

    def __call__(self, data):
        return self._function(self, data)


class FieldInfoContainer(UserDict):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def setup_fluid_index_fields(self):
        def _TranslationFunc(field, data):
            return data["index", "vertex_x"]

        self.add_field(("gas", "vertex_x"), function=_TranslationFunc)

    def add_field(self, name, function):
        self[name] = DerivedField(name, function, ds=self.ds)

    def load_all_plugins(self) -> None:
        setup_fluid_fields(self)
        deps = self.check_derived_fields([])
        self.ds.field_dependencies.update(deps)
        dfl = set(deps.keys())
        self.ds.derived_field_list = sorted(dfl)

    def check_derived_fields(self, fields_to_check=None):
        deps = {}
        if fields_to_check is None:
            fields_to_check = list(self.keys())
        else:
            pass
        for field in fields_to_check:
            fi = self[field]
            try:
                fd = FieldDetector(ds=fi.ds)
                fd[fi.name]
            except Exception:
                self.pop(field)
                continue
            deps[field] = fd
        return deps


class Dataset:
    def __init__(self):
        self.coordinates = CartesianCoordinateHandler()

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
