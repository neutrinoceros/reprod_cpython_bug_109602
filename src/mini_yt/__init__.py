from collections import UserDict, defaultdict

import numpy as np


class Coordinates:
    def setup_fields(self, registry):
        def _vert(field, data):
            return np.random.random((32, 32, 32, 8))

        registry.add_field(("index", "vertex_x"), function=_vert)


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

    def add_field(self, name, function):
        self[name] = DerivedField(name, function, ds=self.ds)

    def check_derived_fields(self):
        deps = {}
        fields_to_check = list(self.keys())
        for field in fields_to_check:
            fi = self[field]
            try:
                fd = FieldDetector(ds=fi.ds)
                fd[fi.name]
            except Exception:
                continue
            deps[field] = fd
        return deps


class Dataset:
    def __init__(self):
        self.field_info = FieldInfoContainer(self)

    def _get_field_info(self, field, /):
        ftype, fname = field
        if (ftype, fname) in self.field_info:
            return self.field_info[ftype, fname]
        else:
            raise Exception
