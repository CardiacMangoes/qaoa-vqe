import hashlib


def hash_list(t):
    assert type(t) is list

    m = hashlib.sha256()
    for i in t:
        m.update(bytes(str(i).encode('utf-8')))
    return m.hexdigest()


def hash_dict(t):
    assert type(t) is dict

    items = sorted(t.items())  # dict_items
    return hash_list(list(items))
