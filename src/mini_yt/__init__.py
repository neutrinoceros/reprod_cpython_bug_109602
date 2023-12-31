from collections import UserDict, defaultdict


class Coordinates:
    def setup_fields(self, registry):
        def _vert(data):
            return list(range(16**3*8))

        registry.add_field("c", function=_vert)


class FieldDetector(defaultdict):
    def __init__(self, ds):
        self.ds = ds
        super().__init__(lambda: 1)

    def __missing__(self, field):
        finfo = self.ds._get_field_info(field)
        vv = finfo(self)
        self[finfo.name] = vv
        return vv


class Field:
    def __init__(self, name, function, ds):
        self.name = name
        self.ds = ds
        self._function = function

    def __call__(self, data):
        return self._function(data)


class FieldInfoContainer(UserDict):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def add_field(self, name, function):
        self[name] = Field(name, function, ds=self.ds)

    def check_fields(self):
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
        if field in self.field_info:
            return self.field_info[field]
        else:
            raise Exception
