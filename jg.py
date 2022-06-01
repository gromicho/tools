def make_iterable( thing ):
    try:
        iter(thing)
    except TypeError:
        return ( thing, )
    else:
        return thing