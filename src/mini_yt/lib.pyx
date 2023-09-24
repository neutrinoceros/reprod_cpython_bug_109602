# distutils: language = c++

def bar(data, field_names):
    units = data[field_names].units
    raise AttributeError