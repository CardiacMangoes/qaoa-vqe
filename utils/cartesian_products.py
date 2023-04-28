import itertools  # Functions creating iterators for efficient looping


def dict_product(dic):
    """
    Makes a cartesian product of a dictionary that has lists for values
    """
    keys = dic.keys()
    values_cp = itertools.product(*dic.values())
    return list(dict(zip(keys, values)) for values in values_cp)
