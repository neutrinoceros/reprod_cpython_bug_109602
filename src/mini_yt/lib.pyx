# distutils: language = c++

def bar(data, field_names, bulk_vector):
    units = data[field_names[0]].units
    raise AttributeError